#All the required packages for running the banking application
from flask import Flask, render_template, request, redirect, session
from flask import url_for, flash, jsonify
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, scoped_session
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import numpy as np
from sklearn.ensemble import IsolationForest

#API Section to connect with the DB
app = Flask(__name__)
app.secret_key = "bank_secret"

#Loading the API key using env (i.e Neon.tech DB)
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///bank.db")
if DATABASE_URL.startswith("postgresql") and "sslmode" not in DATABASE_URL:
    sep = "?" if "?" not in DATABASE_URL else "&"
    DATABASE_URL = f"{DATABASE_URL}{sep}sslmode=require"
engine = create_engine(DATABASE_URL, echo=False, future=True)
Base = declarative_base()
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))

#The User class that takes in their ID, name and etc..

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    pin = Column(String, nullable=False)
    accounts = relationship("Account", back_populates="user", cascade="all, delete-orphan")

#The account class that takes in the required details
class Account(Base):
    __tablename__ = "accounts"
    acc_no = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    balance = Column(Float, default=0.0)
    type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account", cascade="all, delete-orphan")

#The transaction class to handle the transactions smoothly
class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    acc_no = Column(Integer, ForeignKey("accounts.acc_no"))
    type = Column(String)
    amount = Column(Float)
    time = Column(DateTime, default=datetime.utcnow)
    account = relationship("Account", back_populates="transactions")

# ================= ANOMALY DETECTION HELPERS =================
TX_TYPES = ("DEPOSIT", "WITHDRAW", "TRANSFER_IN", "TRANSFER_OUT", "INTEREST")

def _parse_dt(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def _txn_features(tx, prev_dt=None):
    # amount features
    amt = float(tx.amount or 0.0)
    sign = 1.0
    if tx.type in ("WITHDRAW", "TRANSFER_OUT"):
        sign = -1.0
    amt_signed = amt * sign
    log_amt = np.log1p(abs(amt)) if amt > 0 else 0.0
    # time features
    dt = _parse_dt(tx.time) or datetime.utcnow()
    tod = dt.hour + dt.minute/60.0
    tod_sin = np.sin(2*np.pi*(tod/24.0))
    tod_cos = np.cos(2*np.pi*(tod/24.0))
    dow = dt.weekday()  # 0-6
    dow_oh = [1.0 if i==dow else 0.0 for i in range(7)]
    # delta time
    delta_min = 0.0
    if prev_dt:
        delta_min = max(0.0, (dt - prev_dt).total_seconds()/60.0)
        delta_min = np.log1p(delta_min)
    # type one-hot
    t_oh = [1.0 if tx.type == t else 0.0 for t in TX_TYPES]
    return [amt_signed, log_amt, tod_sin, tod_cos, delta_min] + t_oh + dow_oh

def _train_iforest(X):
    if len(X) < 20:
        return None  # too little data, use rule-based fallback
    try:
        model = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
        model.fit(X)
        return model
    except Exception:
        return None

def compute_user_anomalies(dbs, uid):
    # Collect all transactions for user's accounts
    accounts = dbs.query(Account).filter_by(user_id=uid).all()
    all_tx = []
    for a in accounts:
        txs = dbs.query(Transaction).filter_by(acc_no=a.acc_no).order_by(Transaction.time.asc()).all()
        prev = None
        for t in txs:
            all_tx.append((a.acc_no, t, prev))
            prev = _parse_dt(t.time)
    # Build features
    X = [ _txn_features(t, prev) for (_, t, prev) in all_tx ]
    model = _train_iforest(X)
    flags_by_acc = {}
    alerts = []
    if model:
        scores = model.decision_function(X)  # higher is normal; lower is anomalous
        thresh = np.quantile(scores, 0.12)  # flag lowest ~12%
        for i, (acc_no, t, _) in enumerate(all_tx):
            if scores[i] <= thresh:
                flags_by_acc.setdefault(acc_no, set()).add(t.id)
                alerts.append({
                    "acc_no": acc_no,
                    "id": t.id,
                    "type": t.type,
                    "amount": float(t.amount or 0.0),
                    "time": str(t.time or ""),
                    "score": float(scores[i])
                })
    else:
        # Fallback: z-score by type on amount
        by_type = {t: [] for t in TX_TYPES}
        for (_, t, _) in all_tx:
            by_type[t.type].append(float(t.amount or 0.0))
        stats = {}
        for k, arr in by_type.items():
            if arr:
                m = float(np.mean(arr))
                s = float(np.std(arr))
                stats[k] = (m, s if s>0 else 1.0)
        for (acc_no, t, _) in all_tx:
            m, s = stats.get(t.type, (0.0, 1.0))
            z = abs((float(t.amount or 0.0) - m)/s)
            if z >= 2.5:
                flags_by_acc.setdefault(acc_no, set()).add(t.id)
                alerts.append({
                    "acc_no": acc_no,
                    "id": t.id,
                    "type": t.type,
                    "amount": float(t.amount or 0.0),
                    "time": str(t.time or ""),
                    "score": -z
                })
    # Sort alerts by severity (lower score first)
    alerts.sort(key=lambda x: x["score"])
    return flags_by_acc, alerts

def score_hypothetical_transfer(dbs, uid, from_acc, amt):
    # Build training set from user's history
    accounts = dbs.query(Account).filter_by(user_id=uid).all()
    all_tx = []
    for a in accounts:
        txs = dbs.query(Transaction).filter_by(acc_no=a.acc_no).order_by(Transaction.time.asc()).all()
        prev = None
        for t in txs:
            all_tx.append((a.acc_no, t, prev))
            prev = _parse_dt(t.time)
    if not all_tx:
        return {"high": False, "score": 0.0}
    X = [ _txn_features(t, prev) for (_, t, prev) in all_tx ]
    model = _train_iforest(X)
    if not model:
        # Fallback: compare amount to user's typical outgoing amounts
        outs = [float(t.amount or 0.0) for (_, t, _) in all_tx if t.type in ("WITHDRAW","TRANSFER_OUT")]
        if len(outs) < 5:
            return {"high": False, "score": 0.0}
        m = float(np.mean(outs))
        s = float(np.std(outs) or 1.0)
        z = (float(amt) - m)/s
        return {"high": z >= 2.5, "score": -abs(z)}
    # Create hypothetical txn resembling a transfer out from from_acc at current time
    dummy = Transaction(acc_no=from_acc, type="TRANSFER_OUT", amount=float(amt), time=datetime.utcnow())
    # prev time for same account is last txn time
    prev = None
    for acc_no, t, _ in reversed(all_tx):
        if acc_no == from_acc:
            prev = _parse_dt(t.time)
            break
    x = np.array([_txn_features(dummy, prev)])
    score = float(model.decision_function(x)[0])
    # threshold consistent with compute_user_anomalies
    scores_hist = model.decision_function(X)
    thresh = np.quantile(scores_hist, 0.12)
    return {"high": score <= thresh, "score": score}

# ================= DATABASE =================
def get_db():
    return None

def init_db():
    Base.metadata.create_all(engine)
    dbs = SessionLocal()
    try:
        user = dbs.get(User, 1)
        if not user:
            user = User(id=1, name="Harshit", pin=generate_password_hash("1234"))
            dbs.add(user)
            if not dbs.get(Account, 101):
                dbs.add(Account(acc_no=101, user_id=1, balance=0.0, type="SAVINGS"))
            if not dbs.get(Account, 102):
                dbs.add(Account(acc_no=102, user_id=1, balance=0.0, type="CURRENT"))
            dbs.commit()
    finally:
        dbs.close()

# ================= LOGIN =================
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        uid = (request.form.get("uid") or "").strip()
        pin = (request.form.get("pin", "") or "").strip()
        dbs = SessionLocal()
        try:
            if not uid.isdigit():
                flash("Please enter a numeric User ID")
                return render_template("login.html")
            user = dbs.get(User, int(uid))
            ok = False
            if user:
                stored = user.pin or ""
                try:
                    ok = check_password_hash(stored, pin)
                except Exception:
                    ok = stored == pin
            if ok:
                session["uid"] = int(uid)
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid credentials")
        finally:
            dbs.close()
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        pin = request.form.get("pin", "").strip()
        if not name:
            flash("Name is required")
            return render_template("register.html")
        if not pin:
            flash("PIN is required")
            return render_template("register.html")
        if not (pin.isdigit() and len(pin) == 4):
            flash("PIN must be exactly 4 digits")
            return render_template("register.html")
        dbs = SessionLocal()
        try:
            max_id = dbs.query(User).order_by(User.id.desc()).first()
            new_id = (max_id.id + 1) if max_id else 1
            user = User(id=new_id, name=name, pin=generate_password_hash(pin))
            dbs.add(user)
            max_acc = dbs.query(Account).order_by(Account.acc_no.desc()).first()
            base = (max_acc.acc_no + 1) if max_acc else 100
            dbs.add(Account(acc_no=base, user_id=new_id, balance=0.0, type="SAVINGS"))
            dbs.commit()
            flash(f"Registered. Your User ID is {new_id}. Keep your PIN safe.")
            return redirect(url_for("login"))
        finally:
            dbs.close()
    return render_template("register.html")

# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    if "uid" not in session:
        return redirect("/")
    dbs = SessionLocal()
    try:
        accounts = dbs.query(Account).filter_by(user_id=session["uid"]).order_by(Account.acc_no).all()
        total_balance = sum(a.balance for a in accounts)
        flags, alerts = compute_user_anomalies(dbs, session["uid"])  # alerts sorted by severity
        # show top 5 alerts across accounts
        top_alerts = alerts[:5]
        return render_template("dashboard.html", accounts=accounts, total_balance=total_balance, top_alerts=top_alerts)
    finally:
        dbs.close()

# ================= DEPOSIT =================
@app.route("/deposit", methods=["POST"])
def deposit():
    if "uid" not in session:
        return redirect("/")
    acc = int(request.form.get("acc"))
    amt = float(request.form.get("amt", 0) or 0)
    if amt <= 0:
        return redirect(url_for("dashboard"))
    dbs = SessionLocal()
    try:
        a = dbs.get(Account, acc)
        if not a or a.user_id != session["uid"]:
            return redirect(url_for("dashboard"))
        a.balance += amt
        dbs.add(Transaction(acc_no=acc, type="DEPOSIT", amount=amt, time=datetime.utcnow()))
        dbs.commit()
    finally:
        dbs.close()
    return redirect(url_for("dashboard"))

# ================= WITHDRAW =================
@app.route("/withdraw", methods=["POST"])
def withdraw():
    if "uid" not in session:
        return redirect("/")
    acc = int(request.form.get("acc"))
    amt = float(request.form.get("amt", 0) or 0)
    if amt <= 0:
        return redirect(url_for("dashboard"))
    dbs = SessionLocal()
    try:
        a = dbs.get(Account, acc)
        if not a or a.user_id != session["uid"] or a.balance < amt:
            return redirect(url_for("dashboard"))
        a.balance -= amt
        dbs.add(Transaction(acc_no=acc, type="WITHDRAW", amount=amt, time=datetime.utcnow()))
        dbs.commit()
    finally:
        dbs.close()
    return redirect(url_for("dashboard"))

# ================= TRANSFER =================
@app.route("/transfer", methods=["POST"])
def transfer():
    if "uid" not in session:
        return redirect("/")
    from_acc = int(request.form.get("from"))
    to_acc = int(request.form.get("to"))
    amt = float(request.form.get("amt", 0) or 0)
    pin_confirm = (request.form.get("pin_confirm", "") or "").strip()
    if amt <= 0 or from_acc == to_acc:
        return redirect(url_for("dashboard"))
    dbs = SessionLocal()
    try:
        u = dbs.get(User, session["uid"])
        if not u:
            return redirect(url_for("dashboard"))
        try:
            pin_ok = check_password_hash(u.pin or "", pin_confirm)
        except Exception:
            pin_ok = (u.pin or "") == pin_confirm
        if not pin_ok:
            flash("Invalid PIN confirmation for transfer")
            return redirect(url_for("dashboard"))
        fa = dbs.get(Account, from_acc)
        ta = dbs.get(Account, to_acc)
        if not fa or fa.user_id != session["uid"] or not ta or fa.balance < amt:
            return redirect(url_for("dashboard"))
        fa.balance -= amt
        ta.balance += amt
        dbs.add(Transaction(acc_no=from_acc, type="TRANSFER_OUT", amount=amt, time=datetime.utcnow()))
        dbs.add(Transaction(acc_no=to_acc, type="TRANSFER_IN", amount=amt, time=datetime.utcnow()))
        dbs.commit()
    finally:
        dbs.close()
    return redirect(url_for("dashboard"))

# ================= INTEREST =================
@app.route("/interest/<int:acc>")
def interest(acc):
    if "uid" not in session:
        return redirect("/")
    dbs = SessionLocal()
    try:
        a = dbs.get(Account, acc)
        if not a or a.user_id != session["uid"]:
            return redirect(url_for("dashboard"))
        last = (
            dbs.query(Transaction)
            .filter_by(acc_no=acc, type="INTEREST")
            .order_by(Transaction.time.desc())
            .first()
        )
        last_time = None
        if last and last.time:
            last_time = last.time
            if isinstance(last_time, str):
                try:
                    last_time = datetime.fromisoformat(last_time)
                except Exception:
                    last_time = None
        if last_time and (datetime.utcnow() - last_time) < timedelta(days=30):
            flash("Interest can be applied only once every 30 days")
            return redirect(url_for("dashboard"))
        interest_amt = round(a.balance * 0.04, 2)
        a.balance += interest_amt
        dbs.add(Transaction(acc_no=acc, type="INTEREST", amount=interest_amt, time=datetime.utcnow()))
        dbs.commit()
    finally:
        dbs.close()
    return redirect(url_for("dashboard"))

# ================= TRANSACTION HISTORY =================
@app.route("/history/<int:acc>")
def history(acc):
    if "uid" not in session:
        return redirect("/")
    dbs = SessionLocal()
    try:
        a = dbs.get(Account, acc)
        if not a or a.user_id != session["uid"]:
            return redirect(url_for("dashboard"))
        txns = (
            dbs.query(Transaction)
            .filter_by(acc_no=acc)
            .order_by(Transaction.time.desc())
            .all()
        )
        # Compute anomaly flags for this account
        flags, _ = compute_user_anomalies(dbs, session["uid"])  # per-user model
        acc_flags = flags.get(acc, set())
        return render_template("history.html", txns=txns, acc=acc, account=a, suspicious_ids=acc_flags)
    finally:
        dbs.close()

@app.route("/api/risk/transfer")
def risk_transfer():
    if "uid" not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        from_acc = int(request.args.get("from") or 0)
        to_acc = int(request.args.get("to") or 0)
        amt = float(request.args.get("amt") or 0)
    except Exception:
        return jsonify({"error": "bad_request"}), 400
    if amt <= 0 or from_acc == to_acc:
        return jsonify({"high": False, "score": 0.0})
    dbs = SessionLocal()
    try:
        fa = dbs.get(Account, from_acc)
        if not fa or fa.user_id != session["uid"]:
            return jsonify({"error": "forbidden"}), 403
        result = score_hypothetical_transfer(dbs, session["uid"], from_acc, amt)
        return jsonify(result)
    finally:
        dbs.close()

@app.route("/api/chart/<int:acc>")
def chart_data(acc):
    if "uid" not in session:
        return jsonify({"error": "unauthorized"}), 401
    dbs = SessionLocal()
    try:
        a = dbs.get(Account, acc)
        if not a or a.user_id != session["uid"]:
            return jsonify({"error": "forbidden"}), 403
        txns = dbs.query(Transaction).filter_by(acc_no=acc).all()
        buckets = {}
        for t in txns:
            ts = t.time
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = None
            key = ts.strftime("%Y-%m") if ts else datetime.utcnow().strftime("%Y-%m")
            if key not in buckets:
                buckets[key] = {"income": 0.0, "expense": 0.0}
            if t.type in ("DEPOSIT", "TRANSFER_IN", "INTEREST"):
                buckets[key]["income"] += float(t.amount or 0)
            elif t.type in ("WITHDRAW", "TRANSFER_OUT"):
                buckets[key]["expense"] += float(t.amount or 0)
        labels = sorted(buckets.keys())
        income = [round(buckets[m]["income"], 2) for m in labels]
        expense = [round(buckets[m]["expense"], 2) for m in labels]
        return jsonify({"labels": labels, "income": income, "expense": expense})
    finally:
        dbs.close()

@app.route("/pin/generate")
def generate_pin():
    if "uid" not in session:
        return redirect("/")
    import random
    new_pin = str(random.randint(1000, 9999))
    dbs = SessionLocal()
    try:
        u = dbs.get(User, session["uid"])
        if u:
            u.pin = generate_password_hash(new_pin)
            dbs.commit()
            flash(f"Your new PIN is {new_pin}. Please store it securely.")
    finally:
        dbs.close()
    return redirect(url_for("dashboard"))

@app.route("/accounts/create", methods=["POST"])
def create_account():
    if "uid" not in session:
        return redirect("/")
    flash("Opening new accounts from dashboard is disabled")
    return redirect(url_for("dashboard"))

@app.route("/accounts/delete/<int:acc_no>")
def delete_account(acc_no):
    if "uid" not in session:
        return redirect("/")
    dbs = SessionLocal()
    try:
        a = dbs.get(Account, acc_no)
        if a and a.user_id == session["uid"]:
            dbs.delete(a)
            dbs.commit()
    finally:
        dbs.close()
    return redirect(url_for("dashboard"))

# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ================= RUN =================
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
