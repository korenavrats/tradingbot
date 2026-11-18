import time
import json
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# =========================================================================
# 📧 הגדרות מייל (נמשכות אוטומטית מתוך ה-Secrets של גיטהאב)
# =========================================================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "")


def send_email_alert(subject, body):
    try:
        if not SENDER_EMAIL or not SENDER_PASSWORD:
            print("⚠️ פרטי מייל אינם מוגדרים במשתני הסביבה.")
            return
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"📧 מייל נשלח בהצלחה: {subject}")
    except Exception as e:
        print(f"⚠️ שגיאה בשליחת מייל: {e}")


class LiveMultiIntervalBot:
    def __init__(self, portfolio_cash, assets_config, state_file="bot_state.json"):
        self.cash = portfolio_cash
        self.initial_cash = portfolio_cash
        self.assets_config = assets_config
        self.state_file = state_file
        self.active_positions = {}
        self.last_weekly_report_date = None
        self.load_state()

    def save_state(self):
        state = {
            "cash": self.cash,
            "positions": self.active_positions,
            "last_weekly_report_date": self.last_weekly_report_date
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=4)

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.cash = state.get("cash", self.cash)
                    self.active_positions = state.get("positions", {})
                    self.last_weekly_report_date = state.get("last_weekly_report_date", None)
                    print(f"✅ טען מצב קודם. פוזיציות פעילות: {list(self.active_positions.keys())}")
            except Exception as e:
                print(f"⚠️ שגיאה בטעינת מצב: {e}")

    def get_total_portfolio_value(self):
        total = self.cash
        for ticker, pos in self.active_positions.items():
            try:
                latest_close = float(yf.download(ticker, period="1d", interval="1m", progress=False)['Close'].iloc[-1])
                total += pos['shares'] * latest_close
            except:
                total += pos['invested_amount']
        return total

    def fetch_live_data(self, ticker, interval="15m"):
        if interval in ["1m", "2m", "5m", "15m", "30m"]:
            max_period = "60d"
        elif interval in ["1h", "90m"]:
            max_period = "730d"
        else:
            max_period = "max"

        df = yf.download(ticker, period=max_period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def calculate_indicators(self, df):
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['ATR_Pct'] = df['ATR'] / df['Close']
        min_oc = pd.concat([df['Open'], df['Close']], axis=1).min(axis=1)
        max_oc = pd.concat([df['Open'], df['Close']], axis=1).max(axis=1)
        df['Body_Size'] = (max_oc - min_oc) / (df['ATR'] + 1e-9)
        df['Lower_Wick'] = (min_oc - df['Low']) / (df['ATR'] + 1e-9)
        df['Upper_Wick'] = (df['High'] - max_oc) / (df['ATR'] + 1e-9)

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        df['Vol_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Rel'] = df['Volume'] / (df['Vol_MA'] + 1e-9)
        return df

    def train_model_for_asset(self, df, levels_list):
        def get_nearest_distance(price):
            if not levels_list: return 0.0
            return min([abs(price - lvl) / lvl for lvl in levels_list])

        df['Dist_To_Level'] = df['Close'].apply(get_nearest_distance)
        df.dropna(inplace=True)

        features = ['Dist_To_Level', 'ATR_Pct', 'Body_Size', 'Lower_Wick', 'Upper_Wick', 'RSI', 'Volume_Rel']
        df['Target'] = 1
        df['Sample_Weight'] = 0.5

        for i in range(14, len(df) - 5):
            row = df.iloc[i]
            close_p = float(row['Close'])
            nearest_lvl = min(levels_list, key=lambda x: abs(x - close_p))
            dist = abs(close_p - nearest_lvl) / nearest_lvl

            if dist <= 0.008:
                future_window = df.iloc[i + 1: i + 6]
                max_fut = future_window['High'].max()
                min_fut = future_window['Low'].min()

                if close_p >= nearest_lvl:
                    if max_fut >= close_p * 1.02 and min_fut >= nearest_lvl * 0.985:
                        df.iloc[i, df.columns.get_loc('Target')] = 2
                        weight = 4.0 if (row['Lower_Wick'] > 0.4 and row['Volume_Rel'] > 1.1) else 2.0
                        df.iloc[i, df.columns.get_loc('Sample_Weight')] = weight
                elif close_p <= nearest_lvl:
                    if min_fut <= close_p * 0.98 and max_fut >= nearest_lvl * 1.015:
                        df.iloc[i, df.columns.get_loc('Target')] = 0
                        weight = 4.0 if (row['Upper_Wick'] > 0.4 and row['Volume_Rel'] > 1.1) else 2.0
                        df.iloc[i, df.columns.get_loc('Sample_Weight')] = weight

        train_df = df[features].copy()
        train_df['Target'] = df['Target']
        train_df['Weight'] = df['Sample_Weight']
        balanced_train = train_df[train_df['Target'] != 1]
        df_neutral = train_df[train_df['Target'] == 1].sample(frac=0.15, random_state=42)
        balanced_train = pd.concat([balanced_train, df_neutral], axis=0)

        train_data = lgb.Dataset(balanced_train[features], label=balanced_train['Target'],
                                 weight=balanced_train['Weight'])
        params = {'objective': 'multiclass', 'num_class': 3, 'learning_rate': 0.03, 'max_depth': 3, 'verbose': -1,
                  'seed': 42, 'num_threads': 1}
        model = lgb.train(params, train_data, num_boost_round=70)
        return model, features

    def check_weekly_report_trigger(self):
        now = datetime.now()
        if now.weekday() == 5 and now.hour == 12:
            today_str = now.strftime("%Y-%m-%d")
            if self.last_weekly_report_date != today_str:
                self.send_weekly_summary()
                self.last_weekly_report_date = today_str
                self.save_state()

    def run_single_tick(self):
        print(f"\n🔄 הרצת סריקת שווקים: {pd.Timestamp.now()}")
        self.check_weekly_report_trigger()

        for ticker, config in self.assets_config.items():
            levels_list = config['levels']
            interval = config['interval']

            df = self.fetch_live_data(ticker, interval=interval)
            df = self.calculate_indicators(df)

            # 🛡️ הגנה: אם יש פוזיציה פעילה לנכס, הבוט מנהל אותה בלבד ולא פותח עסקאות חדשות עליה!
            if ticker not in self.active_positions:
                model, features = self.train_model_for_asset(df, levels_list)
                last_row = df.iloc[-2]
                close_p = float(last_row['Close'])
                atr_val = float(last_row['ATR'])
                nearest_lvl = min(levels_list, key=lambda x: abs(x - close_p))
                dist = abs(close_p - nearest_lvl) / nearest_lvl

                if dist <= 0.008:
                    feat_vector = df[features].iloc[[-2]]
                    pred_proba = model.predict(feat_vector)[0]

                    direction = 'long' if pred_proba[2] > 0.42 else ('short' if pred_proba[0] > 0.42 else None)
                    if direction == 'long' and close_p < nearest_lvl * 0.998:
                        direction = None
                    elif direction == 'short' and close_p > nearest_lvl * 1.002:
                        direction = None

                    if direction:
                        entry_price = close_p
                        if direction == 'long':
                            initial_sl = min(float(last_row['Low']), nearest_lvl) - (atr_val * 1.0)
                            initial_risk_dlr_per_share = entry_price - initial_sl
                            upper_levels = [lvl for lvl in levels_list if lvl > entry_price]
                            tp = min(upper_levels) if upper_levels else entry_price + (initial_risk_dlr_per_share * 4.0)
                        else:
                            initial_sl = max(float(last_row['High']), nearest_lvl) + (atr_val * 1.0)
                            initial_risk_dlr_per_share = initial_sl - entry_price
                            lower_levels = [lvl for lvl in levels_list if lvl < entry_price]
                            tp = max(lower_levels) if lower_levels else entry_price - (initial_risk_dlr_per_share * 4.0)

                        initial_risk_pct = (initial_risk_dlr_per_share / entry_price) * 100
                        expected_profit_dlr_per_share = abs(tp - entry_price)
                        expected_profit_pct = (expected_profit_dlr_per_share / entry_price) * 100

                        if initial_risk_dlr_per_share > 0 and initial_risk_pct <= 1.2:
                            total_portfolio_value = self.get_total_portfolio_value()
                            position_allocation = total_portfolio_value * 0.025

                            if self.cash < position_allocation:
                                print(f"⚠️ אין מספיק מזומן (${self.cash:.2f}) לפתיחת פוזיציה ב-{ticker}")
                                continue

                            shares_count = position_allocation / entry_price
                            total_risk_dollars = initial_risk_dlr_per_share * shares_count
                            total_expected_profit_dollars = expected_profit_dlr_per_share * shares_count

                            self.active_positions[ticker] = {
                                'dir': direction,
                                'entry': entry_price,
                                'shares': shares_count,
                                'invested_amount': position_allocation,
                                'current_sl': initial_sl,
                                'tp': tp,
                                'sl_history': [round(initial_sl, 2)],
                                'max_price': entry_price,
                                'min_price': entry_price
                            }
                            self.cash -= position_allocation
                            self.save_state()

                            msg = (
                                f"🚀 פתיחת עסקה חדשה ב-Live (נר {interval})!\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"📌 נכס: {ticker} ({direction.upper()})\n"
                                f"💵 מחיר כניסה: ${entry_price:.2f}\n"
                                f"📊 סכום פוזיציה (2.5% מהתיק): ${position_allocation:.2f} ({shares_count:.2f} מניות)\n"
                                f"🛡️ סטופ: ${initial_sl:.2f} (הפסד מקסימלי: ${total_risk_dollars:.2f} | {initial_risk_pct:.2f}%)\n"
                                f"🎯 יעד רווח (TP): ${tp:.2f} (רווח צפוי: ${total_expected_profit_dollars:.2f} | {expected_profit_pct:.2f}%)\n"
                            )
                            print(msg)
                            send_email_alert(f"🚨 New Trade: {ticker} ({direction.upper()})", msg)

            else:
                # 🛡️ ניהול פוזיציה קיימת בלבד (סטופ, טריילינג, TP)
                pos = self.active_positions[ticker]
                latest_candle = df.iloc[-1]
                high_val, low_val = float(latest_candle['High']), float(latest_candle['Low'])

                pos['max_price'] = max(pos['max_price'], high_val)
                pos['min_price'] = min(pos['min_price'], low_val)

                sl_changed = False
                exit_triggered = False
                exit_reason = ""
                exit_price = 0

                BREAK_EVEN_TRIGGER = 0.020
                TRAILING_TRIGGER = 0.030
                TRAILING_OFFSET = 0.99

                if pos['dir'] == 'long':
                    if pos['max_price'] >= pos['entry'] * (1 + BREAK_EVEN_TRIGGER) and pos['current_sl'] < pos['entry']:
                        pos['current_sl'] = pos['entry']
                        sl_changed = True
                    if pos['max_price'] >= pos['entry'] * (1 + TRAILING_TRIGGER):
                        new_trail = pos['max_price'] * TRAILING_OFFSET
                        if new_trail > pos['current_sl']:
                            pos['current_sl'] = new_trail
                            sl_changed = True

                    if sl_changed and pos['sl_history'][-1] != round(pos['current_sl'], 2):
                        pos['sl_history'].append(round(pos['current_sl'], 2))
                        self.save_state()
                        msg = f"🛡️ עדכון סטופ-לוס ב-{ticker}!\nסטופ חדש: ${round(pos['current_sl'], 2)}\nהיסטוריה: {pos['sl_history']}"
                        print(msg)
                        send_email_alert(f"⚠️ SL Update: {ticker}", msg)

                    if low_val <= pos['current_sl']:
                        exit_triggered = True
                        exit_price = pos['current_sl']
                        exit_reason = "Trailing / Break-Even / SL Hit"
                    elif high_val >= pos['tp']:
                        exit_triggered = True
                        exit_price = pos['tp']
                        exit_reason = "TP Target Reached"

                else:
                    if pos['min_price'] <= pos['entry'] * (1 - BREAK_EVEN_TRIGGER) and pos['current_sl'] > pos['entry']:
                        pos['current_sl'] = pos['entry']
                        sl_changed = True
                    if pos['min_price'] <= pos['entry'] * (1 - TRAILING_TRIGGER):
                        new_trail = pos['min_price'] * (2 - TRAILING_OFFSET)
                        if new_trail < pos['current_sl']:
                            pos['current_sl'] = new_trail
                            sl_changed = True

                    if sl_changed and pos['sl_history'][-1] != round(pos['current_sl'], 2):
                        pos['sl_history'].append(round(pos['current_sl'], 2))
                        self.save_state()
                        msg = f"🛡️ עדכון סטופ-לוס שורט ב-{ticker}!\nסטופ חדש: ${round(pos['current_sl'], 2)}\nהיסטוריה: {pos['sl_history']}"
                        print(msg)
                        send_email_alert(f"⚠️ SL Update (Short): {ticker}", msg)

                    if high_val >= pos['current_sl']:
                        exit_triggered = True
                        exit_price = pos['current_sl']
                        exit_reason = "Trailing / Break-Even / SL Hit"
                    elif low_val <= pos['tp']:
                        exit_triggered = True
                        exit_price = pos['tp']
                        exit_reason = "TP Target Reached"

                if exit_triggered:
                    pnl_pct = (exit_price - pos['entry']) / pos['entry'] * 100 if pos['dir'] == 'long' else (pos[
                                                                                                                 'entry'] - exit_price) / \
                                                                                                            pos[
                                                                                                                'entry'] * 100
                    pnl_dollars = pos['shares'] * (exit_price - pos['entry']) if pos['dir'] == 'long' else pos[
                                                                                                               'shares'] * (
                                                                                                                       pos[
                                                                                                                           'entry'] - exit_price)

                    self.cash += pos['invested_amount'] + pnl_dollars
                    del self.active_positions[ticker]
                    self.save_state()

                    msg = (
                        f"🏁 עסקה נסגרה ב-Live!\n"
                        f"📌 נכס: {ticker}\n"
                        f"📝 סיבה: {exit_reason}\n"
                        f"💵 מחיר כניסה: ${pos['entry']:.2f} ➔ יציאה: ${exit_price:.2f}\n"
                        f"💰 רווח/הפסד פוזיציה: ${pnl_dollars:+.2f} ({pnl_pct:+.2f}%)\n"
                        f"🏦 מזומן חדש בתיק: ${self.cash:.2f}"
                    )
                    print(msg)
                    send_email_alert(f"🎯 Trade Closed: {ticker} ({pnl_pct:+.2f}%)", msg)

    def send_weekly_summary(self):
        total_portfolio_value = self.cash
        active_details = ""

        for ticker, pos in self.active_positions.items():
            interval = self.assets_config[ticker]['interval']
            try:
                current_price = float(self.fetch_live_data(ticker, interval=interval)['Close'].iloc[-1])
            except:
                current_price = pos['entry']

            curr_pnl_dlr = (current_price - pos['entry']) * pos['shares'] if pos['dir'] == 'long' else (pos[
                                                                                                            'entry'] - current_price) * \
                                                                                                       pos['shares']
            curr_pnl_pct = (curr_pnl_dlr / pos['invested_amount']) * 100
            total_portfolio_value += pos['invested_amount'] + curr_pnl_dlr

            active_details += (
                f"• {ticker} ({pos['dir'].upper()} - נר {interval}):\n"
                f"  - מחיר כניסה: ${pos['entry']:.2f} | נוכחי: ${current_price:.2f}\n"
                f"  - סטופ נוכחי: ${pos['current_sl']:.2f} | היסטוריה: {pos['sl_history']}\n"
                f"  - PnL נוכחי: ${curr_pnl_dlr:+.2f} ({curr_pnl_pct:+.2f}%)\n\n"
            )

        total_return_pct = ((total_portfolio_value - self.initial_cash) / self.initial_cash) * 100
        total_return_dlr = total_portfolio_value - self.initial_cash

        summary_msg = (
            f"📅 דו''ח סיכום שבועי (שבת 12:00)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏦 שווי תיק כולל: ${total_portfolio_value:.2f}\n"
            f"💵 מזומן פנוי בתיק: ${self.cash:.2f}\n"
            f"📈 רווח/הפסד כולל: ${total_return_dlr:+.2f} ({total_return_pct:+.2f}%)\n\n"
            f"📌 פוזיציות פעילות:\n"
            f"{active_details if active_details else 'אין פוזיציות פעילות כרגע.'}"
        )
        print(summary_msg)
        send_email_alert("📅 Weekly Portfolio Performance Report", summary_msg)


if __name__ == "__main__":
    MY_PORTFOLIO_CASH = 1000.0  # סכום תיק התחלתי

    # 🎛️ רשימת המניות שלך וההגדרות המלאות:
    ASSETS_TO_TRADE = {
        "QQQ": {"levels": [697.39, 679.84, 660.00, 651.35], "interval": "15m"},
        "AZ": {"levels": [5.36, 5.95, 6.34], "interval": "15m"},
        "BATRK": {"levels": [49.37, 50.38, 51.36, 52.96, 54.16], "interval": "15m"},
        "XPOF": {"levels": [4.37, 4.48, 4.8, 5.08], "interval": "15m"},
        "BBIO": {"levels": [73.57, 76.94, 79.91], "interval": "15m"},
        "BIPS": {"levels": [36.86, 37.98, 39.08, 39.82], "interval": "15m"},
        "CLOV": {"levels": [4.18, 4.52, 4.86], "interval": "15m"},
        "CSGP": {"levels": [26.78, 28.8, 31.65, 34.38], "interval": "1h"}
    }

    bot = LiveMultiIntervalBot(portfolio_cash=MY_PORTFOLIO_CASH, assets_config=ASSETS_TO_TRADE)
    bot.run_single_tick()