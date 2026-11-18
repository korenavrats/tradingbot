import json
import os
import smtplib
import warnings
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

IL_TZ = ZoneInfo('Asia/Jerusalem')

# =========================================================================
# 📧 הגדרות מייל (נמשכות אוטומטית מתוך ה-Secrets של גיטהאב)
# =========================================================================
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECEIVER_EMAIL = os.environ.get('RECEIVER_EMAIL', '')

# =========================================================================
# ⚙️ פרמטרים
# =========================================================================
POSITION_PCT = 0.025          # גודל פוזיציה מהתיק
MAX_OPEN_POSITIONS = 6        # תקרת חשיפה: 6 * 2.5% = 15%
MAX_RISK_PCT = 1.2            # סיכון התחלתי מקסימלי (%)
MIN_RR = 2.5                  # יחס סיכון/סיכוי מינימלי (0 = ביטול)
PROB_THRESHOLD = 0.42
LEVEL_PROXIMITY = 0.008
MIN_SIGNAL_SAMPLES = 8        # מינימום דוגמאות חיוביות לאימון, אחרת לא נכנסים
COST_PER_SIDE_PCT = 0.0005    # עמלה+החלקה משוערת לכל צד (0.05%)
FRESHNESS_INTERVALS = 2       # נר אחרון חייב להיסגר לפני עד 2 אינטרוולים
BREAK_EVEN_TRIGGER = 0.020
TRAILING_TRIGGER = 0.030
TRAILING_OFFSET = 0.99

INTERVAL_MINUTES = {
    '1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30,
    '60m': 60, '1h': 60, '90m': 90,
}


def send_email_alert(subject, body):
  try:
    if not SENDER_EMAIL or not SENDER_PASSWORD:
      print('⚠️ פרטי מייל אינם מוגדרים במשתני הסביבה.')
      return
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    server.quit()
    print(f'📧 מייל נשלח בהצלחה: {subject}')
  except Exception as e:
    print(f'⚠️ שגיאה בשליחת מייל: {e}')


def dist_to_level(price, levels):
  if not levels:
    return 0.0
  return min(abs(price - lvl) / lvl for lvl in levels)


def position_value(pos, price):
  """שווי פוזיציה נוכחי (כולל לונג ושורט): סכום שהושקע + PnL לא ממומש."""
  if pos['dir'] == 'long':
    pnl = (price - pos['entry']) * pos['shares']
  else:
    pnl = (pos['entry'] - price) * pos['shares']
  return pos['invested_amount'] + pnl, pnl


class LiveMultiIntervalBot:

  def __init__(self, portfolio_cash, assets_config, state_file='bot_state.json'):
    self.cash = portfolio_cash
    self.initial_cash = portfolio_cash
    self.assets_config = assets_config
    self.state_file = state_file
    self.active_positions = {}
    self.last_weekly_report_date = None
    self.last_prices = {}
    self.load_state()

  # ------------------------------------------------------------------ state
  def save_state(self):
    state = {
        'cash': self.cash,
        'positions': self.active_positions,
        'last_weekly_report_date': self.last_weekly_report_date,
    }
    temp_filename = self.state_file + '.tmp'
    try:
      clean_state = json.loads(
          json.dumps(
              state,
              default=lambda x: float(x) if hasattr(x, 'item') else str(x),
          )
      )
      with open(temp_filename, 'w') as f:
        json.dump(clean_state, f, indent=4)
      os.replace(temp_filename, self.state_file)
    except Exception as e:
      print(f'⚠️ שגיאה בשמירת מצב אטומי: {e}')

  def load_state(self):
    if os.path.exists(self.state_file):
      try:
        with open(self.state_file, 'r') as f:
          state = json.load(f)
        self.cash = state.get('cash', self.cash)
        self.active_positions = state.get('positions', {})
        self.last_weekly_report_date = state.get('last_weekly_report_date')
        print(
            '✅ טען מצב קודם. פוזיציות פעילות:'
            f' {list(self.active_positions.keys())}'
        )
      except Exception as e:
        print(f'⚠️ שגיאה בטעינת מצב: {e}')

  # ------------------------------------------------------------------- data
  def fetch_live_data(self, ticker, interval='15m'):
    try:
      if interval in ['1m', '2m', '5m', '15m', '30m']:
        max_period = '60d'
      elif interval in ['1h', '60m', '90m']:
        max_period = '730d'
      else:
        max_period = 'max'

      df = yf.download(
          ticker, period=max_period, interval=interval, progress=False
      )
      if df is None or df.empty:
        return None
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col for col in df.columns
        ]
      df = df.loc[:, ~df.columns.duplicated()]
      df.dropna(inplace=True)
      idx = pd.to_datetime(df.index)
      # נרמול כל האינדקס ל-UTC כדי למנוע בעיות אזורי זמן
      idx = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
      df.index = idx
      return df if len(df) > 20 else None
    except Exception as e:
      print(f'❌ שגיאה בשליפת נתונים עבור {ticker}: {e}')
      return None

  @staticmethod
  def closed_candles(df, interval, now_utc):
    """מחזיר רק נרות שנסגרו בפועל (זמן פתיחה + אורך נר <= עכשיו)."""
    td = pd.Timedelta(minutes=INTERVAL_MINUTES.get(interval, 15))
    return df[df.index + td <= now_utc]

  def calculate_indicators(self, df, levels_list):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    df['ATR_Pct'] = df['ATR'] / df['Close']
    min_oc = pd.concat([df['Open'], df['Close']], axis=1).min(axis=1)
    max_oc = pd.concat([df['Open'], df['Close']], axis=1).max(axis=1)
    df['Body_Size'] = (max_oc - min_oc) / (df['ATR'] + 1e-9)
    df['Lower_Wick'] = (min_oc - df['Low']) / (df['ATR'] + 1e-9)
    df['Upper_Wick'] = (df['High'] - max_oc) / (df['ATR'] + 1e-9)

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Rel'] = df['Volume'] / (df['Vol_MA'] + 1e-9)

    # הפיצ'ר הזה חייב להיות ב-df עצמו כדי שגם החיזוי ישתמש בו
    df['Dist_To_Level'] = df['Close'].apply(
        lambda p: dist_to_level(float(p), levels_list)
    )
    return df

  # ------------------------------------------------------------------ model
  FEATURES = [
      'Dist_To_Level',
      'ATR_Pct',
      'Body_Size',
      'Lower_Wick',
      'Upper_Wick',
      'RSI',
      'Volume_Rel',
  ]

  def train_model_for_asset(self, df, levels_list):
    """מחזיר (model, features, n_signals) או None אם אין מספיק דוגמאות."""
    features = self.FEATURES
    work = df.dropna(subset=features).copy()
    work['Target'] = 1
    work['Sample_Weight'] = 0.5

    t_col = work.columns.get_loc('Target')
    w_col = work.columns.get_loc('Sample_Weight')

    for i in range(14, len(work) - 5):
      row = work.iloc[i]
      close_p = float(row['Close'])
      nearest_lvl = min(levels_list, key=lambda x: abs(x - close_p))
      dist = abs(close_p - nearest_lvl) / nearest_lvl

      if dist <= LEVEL_PROXIMITY:
        future_window = work.iloc[i + 1: i + 6]
        max_fut = future_window['High'].max()
        min_fut = future_window['Low'].min()

        if close_p >= nearest_lvl:
          if max_fut >= close_p * 1.02 and min_fut >= nearest_lvl * 0.985:
            work.iloc[i, t_col] = 2
            strong = row['Lower_Wick'] > 0.4 and row['Volume_Rel'] > 1.1
            work.iloc[i, w_col] = 4.0 if strong else 2.0
        elif close_p <= nearest_lvl:
          if min_fut <= close_p * 0.98 and max_fut >= nearest_lvl * 1.015:
            work.iloc[i, t_col] = 0
            strong = row['Upper_Wick'] > 0.4 and row['Volume_Rel'] > 1.1
            work.iloc[i, w_col] = 4.0 if strong else 2.0

    train_df = work[features].copy()
    train_df['Target'] = work['Target']
    train_df['Weight'] = work['Sample_Weight']

    signals = train_df[train_df['Target'] != 1]
    neutral = train_df[train_df['Target'] == 1]
    if len(signals) < MIN_SIGNAL_SAMPLES:
      return None
    if len(neutral) > 0:
      neutral = neutral.sample(frac=0.15, random_state=42)
    balanced_train = pd.concat([signals, neutral], axis=0)

    train_data = lgb.Dataset(
        balanced_train[features],
        label=balanced_train['Target'],
        weight=balanced_train['Weight'],
    )
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'learning_rate': 0.03,
        'max_depth': 3,
        'verbose': -1,
        'seed': 42,
        'num_threads': 1,
    }
    model = lgb.train(params, train_data, num_boost_round=70)
    return model, features, len(signals)

  # --------------------------------------------------------------- reporting
  def check_weekly_report_trigger(self):
    """דו"ח פעם בשבוע: שבת מ-12:00 שעון ישראל (או ראשון, אם השבת הוחמצה)."""
    try:
      now_il = datetime.now(IL_TZ)
      is_report_time = (now_il.weekday() == 5 and now_il.hour >= 12) or (
          now_il.weekday() == 6
      )
      if not is_report_time:
        return
      iso = now_il.isocalendar()
      week_key = f'{iso[0]}-W{iso[1]:02d}'
      if self.last_weekly_report_date != week_key:
        self.send_weekly_summary()
        self.last_weekly_report_date = week_key
        self.save_state()
    except Exception as e:
      print(f'⚠️ שגיאה בבדיקת דוח שבועי: {e}')

  # ----------------------------------------------------------------- entries
  def look_for_entry(self, ticker, config, df_closed, now_utc):
    levels_list = config['levels']
    interval = config['interval']
    td = pd.Timedelta(minutes=INTERVAL_MINUTES.get(interval, 15))

    if len(self.active_positions) >= MAX_OPEN_POSITIONS:
      return

    last_ts = df_closed.index[-1]
    if now_utc - (last_ts + td) > td * FRESHNESS_INTERVALS:
      print(f'😴 {ticker}: הנר האחרון ישן ({last_ts}), שוק סגור. אין כניסה.')
      return

    last_row = df_closed.iloc[-1]
    if last_row[self.FEATURES].isna().any():
      return

    close_p = float(last_row['Close'])
    nearest_lvl = min(levels_list, key=lambda x: abs(x - close_p))
    dist = abs(close_p - nearest_lvl) / nearest_lvl
    if dist > LEVEL_PROXIMITY:
      return

    trained = self.train_model_for_asset(df_closed, levels_list)
    if trained is None:
      print(f'ℹ️ {ticker}: מעט מדי דוגמאות לאימון, מדלג.')
      return
    model, features, _ = trained

    pred_proba = model.predict(df_closed[features].iloc[[-1]])[0]
    atr_val = float(last_row['ATR'])

    direction = (
        'long'
        if pred_proba[2] > PROB_THRESHOLD
        else ('short' if pred_proba[0] > PROB_THRESHOLD else None)
    )
    if direction == 'long' and close_p < nearest_lvl * 0.998:
      direction = None
    elif direction == 'short' and close_p > nearest_lvl * 1.002:
      direction = None
    if not direction:
      return

    entry_price = close_p
    if direction == 'long':
      initial_sl = min(float(last_row['Low']), nearest_lvl) - atr_val
      risk_per_share = entry_price - initial_sl
      upper = [lvl for lvl in levels_list if lvl > entry_price]
      tp = min(upper) if upper else entry_price + risk_per_share * 4.0
    else:
      initial_sl = max(float(last_row['High']), nearest_lvl) + atr_val
      risk_per_share = initial_sl - entry_price
      lower = [lvl for lvl in levels_list if lvl < entry_price]
      tp = max(lower) if lower else entry_price - risk_per_share * 4.0

    if risk_per_share <= 0:
      return
    initial_risk_pct = risk_per_share / entry_price * 100
    profit_per_share = abs(tp - entry_price)
    expected_profit_pct = profit_per_share / entry_price * 100
    if initial_risk_pct > MAX_RISK_PCT:
      return
    if profit_per_share / risk_per_share < MIN_RR:
      print(f'ℹ️ {ticker}: יחס סיכון/סיכוי נמוך מדי, מדלג.')
      return

    total_value = self.get_total_portfolio_value()
    allocation = total_value * POSITION_PCT
    if self.cash < allocation:
      print(f'⚠️ אין מספיק מזומן (${self.cash:.2f}) לפתיחת פוזיציה ב-{ticker}')
      return

    shares = allocation / entry_price
    total_risk = risk_per_share * shares
    total_profit = profit_per_share * shares

    self.active_positions[ticker] = {
        'dir': direction,
        'entry': entry_price,
        'shares': shares,
        'invested_amount': allocation,
        'current_sl': initial_sl,
        'tp': tp,
        'sl_history': [round(initial_sl, 2)],
        'max_price': entry_price,
        'min_price': entry_price,
        'last_checked_timestamp': last_ts.isoformat(),
    }
    self.cash -= allocation
    self.save_state()

    msg = (
        f'🚀 פתיחת עסקה חדשה ב-Live (נר {interval})!\n'
        f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        f'📌 נכס: {ticker} ({direction.upper()})\n'
        f'💵 מחיר כניסה: ${entry_price:.2f}\n'
        f'📊 סכום פוזיציה ({POSITION_PCT * 100:.1f}% מהתיק): ${allocation:.2f}'
        f' ({shares:.2f} מניות)\n'
        f'🛡️ סטופ: ${initial_sl:.2f} (הפסד מקסימלי: ${total_risk:.2f} |'
        f' {initial_risk_pct:.2f}%)\n'
        f'🎯 יעד רווח (TP): ${tp:.2f} (רווח צפוי: ${total_profit:.2f} |'
        f' {expected_profit_pct:.2f}%)\n'
    )
    print(msg)
    send_email_alert(f'🚨 New Trade: {ticker} ({direction.upper()})', msg)

  # --------------------------------------------------------------- management
  def manage_position(self, ticker, df_closed):
    pos = self.active_positions[ticker]

    new_candles = None
    last_checked = pos.get('last_checked_timestamp')
    if last_checked:
      try:
        last_dt = pd.to_datetime(last_checked, utc=True)
        new_candles = df_closed[df_closed.index > last_dt]
      except Exception:
        new_candles = None
    if new_candles is None:
      new_candles = df_closed.tail(2)

    exit_price = None
    exit_reason = ''
    sl_start = pos['current_sl']

    for ts, candle in new_candles.iterrows():
      o = float(candle['Open'])
      h = float(candle['High'])
      l = float(candle['Low'])

      # 1) קודם בודקים את הסטופ/TP הקיימים מול הנר (לפני כל הזזה),
      #    ובמקרה שנפגעו שניהם באותו נר מניחים סטופ (שמרני).
      if pos['dir'] == 'long':
        if l <= pos['current_sl']:
          exit_price = min(o, pos['current_sl'])  # גאפ מטה = יציאה במחיר הפתיחה
          exit_reason = 'Stop Hit (SL / Break-Even / Trailing)'
          break
        if h >= pos['tp']:
          exit_price = max(o, pos['tp'])
          exit_reason = 'TP Target Reached'
          break
        # 2) רק אחר כך מעדכנים max ומזיזים סטופ לנר הבא
        pos['max_price'] = max(pos['max_price'], h)
        if (
            pos['max_price'] >= pos['entry'] * (1 + BREAK_EVEN_TRIGGER)
            and pos['current_sl'] < pos['entry']
        ):
          pos['current_sl'] = pos['entry']
        if pos['max_price'] >= pos['entry'] * (1 + TRAILING_TRIGGER):
          new_trail = pos['max_price'] * TRAILING_OFFSET
          if new_trail > pos['current_sl']:
            pos['current_sl'] = new_trail
      else:
        if h >= pos['current_sl']:
          exit_price = max(o, pos['current_sl'])
          exit_reason = 'Stop Hit (SL / Break-Even / Trailing)'
          break
        if l <= pos['tp']:
          exit_price = min(o, pos['tp'])
          exit_reason = 'TP Target Reached'
          break
        pos['min_price'] = min(pos['min_price'], l)
        if (
            pos['min_price'] <= pos['entry'] * (1 - BREAK_EVEN_TRIGGER)
            and pos['current_sl'] > pos['entry']
        ):
          pos['current_sl'] = pos['entry']
        if pos['min_price'] <= pos['entry'] * (1 - TRAILING_TRIGGER):
          new_trail = pos['min_price'] * (2 - TRAILING_OFFSET)
          if new_trail < pos['current_sl']:
            pos['current_sl'] = new_trail

      if pos['sl_history'][-1] != round(pos['current_sl'], 2):
        pos['sl_history'].append(round(pos['current_sl'], 2))

    if not df_closed.empty:
      pos['last_checked_timestamp'] = df_closed.index[-1].isoformat()

    if exit_price is None:
      self.save_state()
      if pos['current_sl'] != sl_start:
        side = ' שורט' if pos['dir'] == 'short' else ''
        sl_msg = (
            f'🛡️ עדכון סטופ-לוס{side} ב-{ticker}!\n'
            f'סטופ חדש: ${pos["current_sl"]:.2f}\n'
            f'היסטוריה: {pos["sl_history"]}'
        )
        print(sl_msg)
        send_email_alert(f'⚠️ SL Update: {ticker} ({pos["dir"].upper()})', sl_msg)
      return

    # --- סגירת עסקה
    if pos['dir'] == 'long':
      gross = pos['shares'] * (exit_price - pos['entry'])
    else:
      gross = pos['shares'] * (pos['entry'] - exit_price)
    costs = pos['invested_amount'] * COST_PER_SIDE_PCT * 2
    pnl_dollars = gross - costs
    pnl_pct = pnl_dollars / pos['invested_amount'] * 100

    self.cash += pos['invested_amount'] + pnl_dollars
    del self.active_positions[ticker]
    self.save_state()

    msg = (
        f'🏁 עסקה נסגרה ב-Live!\n'
        f'📌 נכס: {ticker} ({pos["dir"].upper()})\n'
        f'📝 סיבה: {exit_reason}\n'
        f'💵 מחיר כניסה: ${pos["entry"]:.2f} ➔ יציאה: ${exit_price:.2f}\n'
        f'💰 רווח/הפסד נטו (אחרי עלויות משוערות): ${pnl_dollars:+.2f}'
        f' ({pnl_pct:+.2f}% מהפוזיציה)\n'
        f'🏦 מזומן חדש בתיק: ${self.cash:.2f}'
    )
    print(msg)
    send_email_alert(f'🎯 Trade Closed: {ticker} ({pnl_pct:+.2f}%)', msg)

  # -------------------------------------------------------------------- main
  def run_single_tick(self):
    now_utc = pd.Timestamp.now(tz='UTC')
    print(f'\n🔄 הרצת סריקת שווקים (שעון ישראל): {datetime.now(IL_TZ)}')

    self.check_weekly_report_trigger()

    for ticker, config in self.assets_config.items():
      try:
        interval = config['interval']
        df = self.fetch_live_data(ticker, interval=interval)
        if df is None:
          print(f'⚠️ דילוג על {ticker} עקב כשל בשליפת נתונים.')
          continue

        self.last_prices[ticker] = float(df['Close'].iloc[-1])

        df_closed = self.closed_candles(df, interval, now_utc)
        if len(df_closed) < 40:
          print(f'⚠️ {ticker}: אין מספיק נרות סגורים.')
          continue
        df_closed = self.calculate_indicators(
            df_closed.copy(), config['levels']
        )

        if ticker in self.active_positions:
          self.manage_position(ticker, df_closed)
        else:
          self.look_for_entry(ticker, config, df_closed, now_utc)

      except Exception as asset_err:
        print(f'❌ שגיאה בטיפול בנכס {ticker}: {asset_err}')
      finally:
        # שמירה אחרי כל נכס, גם אם הייתה שגיאה באמצע
        self.save_state()

  # ----------------------------------------------------------------- values
  def get_total_portfolio_value(self):
    total = self.cash
    for ticker, pos in self.active_positions.items():
      price = self.last_prices.get(ticker, pos['entry'])
      value, _ = position_value(pos, price)
      total += value
    return total

  def send_weekly_summary(self):
    total_portfolio_value = self.cash
    active_details = ''

    for ticker, pos in self.active_positions.items():
      interval = self.assets_config[ticker]['interval']
      current_price = pos['entry']
      try:
        df = self.fetch_live_data(ticker, interval=interval)
        if df is not None:
          current_price = float(df['Close'].iloc[-1])
      except Exception as e:
        print(f'⚠️ שגיאה בשליפת מחיר לדו"ח שבועי עבור {ticker}: {e}')

      value, curr_pnl_dlr = position_value(pos, current_price)
      curr_pnl_pct = curr_pnl_dlr / pos['invested_amount'] * 100
      total_portfolio_value += value

      active_details += (
          f"• {ticker} ({pos['dir'].upper()} - נר {interval}):\n"
          f"  - מחיר כניסה: ${pos['entry']:.2f} | נוכחי: ${current_price:.2f}\n"
          f"  - סטופ נוכחי: ${pos['current_sl']:.2f} | היסטוריה:"
          f" {pos['sl_history']}\n"
          f'  - PnL נוכחי: ${curr_pnl_dlr:+.2f} ({curr_pnl_pct:+.2f}%)\n\n'
      )

    total_return_dlr = total_portfolio_value - self.initial_cash
    total_return_pct = total_return_dlr / self.initial_cash * 100

    summary_msg = (
        f"📅 דו''ח סיכום שבועי (שבת 12:00)\n"
        f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        f'🏦 שווי תיק כולל: ${total_portfolio_value:.2f}\n'
        f'💵 מזומן פנוי בתיק: ${self.cash:.2f}\n'
        f'📈 רווח/הפסד כולל: ${total_return_dlr:+.2f}'
        f' ({total_return_pct:+.2f}%)\n\n'
        f'📌 פוזיציות פעילות:\n'
        f"{active_details if active_details else 'אין פוזיציות פעילות כרגע.'}"
    )
    print(summary_msg)
    send_email_alert('📅 Weekly Portfolio Performance Report', summary_msg)


if __name__ == '__main__':
  MY_PORTFOLIO_CASH = 1000.0  # סכום תיק התחלתי

  ASSETS_TO_TRADE = {
      'QQQ': {'levels': [697.39, 679.84, 660.00, 651.35], 'interval': '15m'},
      'AZ': {'levels': [5.36, 5.95, 6.34], 'interval': '15m'},
      'BATRK': {'levels': [49.37, 50.38, 51.36, 52.96, 54.16], 'interval': '15m'},
      'XPOF': {'levels': [4.37, 4.48, 4.8, 5.08], 'interval': '15m'},
      'BBIO': {'levels': [73.57, 76.94, 79.91], 'interval': '15m'},
      'CLOV': {'levels': [4.18, 4.52, 4.86], 'interval': '15m'},
      'CSGP': {'levels': [26.78, 28.8, 31.65, 34.38], 'interval': '1h'},
      'DAL': {'levels': [86.10, 84], 'interval': '15m'},
  }

  bot = LiveMultiIntervalBot(
      portfolio_cash=MY_PORTFOLIO_CASH, assets_config=ASSETS_TO_TRADE
  )
  bot.run_single_tick()
