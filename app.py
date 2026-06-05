"""
Project Intelligence Chatbot — Streamlit + Claude AI (AIML)
Dataset: users_large_dataset.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import re
import os
from datetime import datetime

# ─── Helper Functions ────────────────────────────────────────────────────────
def df_to_markdown(df) -> str:
    """Helper to convert a pandas DataFrame to a markdown table without tabulate dependency."""
    if df is None or df.empty:
        return ""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, row in df.iterrows():
        row_str = [str(val).replace("\n", " ") for val in row.values]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Intelligence Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Chat container */
.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding: 12px;
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 16px;
}

/* User bubble */
.user-msg {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px auto;
    max-width: 75%;
    font-size: 0.93rem;
    box-shadow: 0 4px 12px rgba(102,126,234,0.3);
}

/* Bot bubble */
.bot-msg {
    background: rgba(255,255,255,0.07);
    color: #e8eaf6;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px auto 6px 0;
    max-width: 80%;
    font-size: 0.93rem;
    border: 1px solid rgba(255,255,255,0.12);
}

/* Stat cards */
.stat-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    transition: transform .2s;
}
.stat-card:hover { transform: translateY(-3px); }
.stat-value { font-size: 2rem; font-weight: 700; color: #7986cb; }
.stat-label { font-size: 0.8rem; color: #9e9e9e; margin-top: 4px; }

/* Sidebar */
.css-1d391kg { background: rgba(26,26,46,0.95) !important; }

/* Input box */
.stTextInput input {
    background: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border: 2px solid #667eea !important;
    border-radius: 12px !important;
}
div[data-baseweb="input"] {
    background: #ffffff !important;
    border-radius: 12px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #4a4a7a; border-radius: 4px; }

/* Table */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("users_large_dataset.csv")
    rows = []
    for _, user in df.iterrows():
        try:
            projs = json.loads(user["projects"])
        except:
            continue
        for p in projs:
            tasks = p.get("tasks", [])
            total_tasks   = len(tasks)
            completed     = sum(1 for t in tasks if t.get("progress", 0) >= 100)
            avg_prog      = np.mean([t.get("progress", 0) for t in tasks]) if tasks else 0
            pc = pd.to_datetime(p.get("createdAt"), errors="coerce")
            pu = pd.to_datetime(p.get("updatedAt"), errors="coerce")
            end_dates = [pd.to_datetime(t.get("endDate"), errors="coerce")
                         for t in tasks
                         if pd.notnull(pd.to_datetime(t.get("endDate"), errors="coerce"))]
            planned_end = max(end_dates) if end_dates else None
            rows.append({
                "user_id": user["id"], "username": user["username"],
                "email": user["email"], "dob": user["dob"],
                "is_verified": user["isVerified"], "is_deleted_user": user["isDeleted"],
                "project_id": p["id"], "project_name": p["projectName"],
                "is_deleted": p.get("isDeleted", False),
                "proj_created": pc, "proj_updated": pu,
                "planned_end": planned_end,
                "total_tasks": total_tasks, "completed_tasks": completed,
                "avg_progress": avg_prog,
                "tasks": tasks,
            })
    proj_df = pd.DataFrame(rows)
    NOW = pd.Timestamp("2026-11-15")
    proj_df["project_age_days"]   = (NOW - proj_df["proj_created"]).dt.days.fillna(0)
    proj_df["days_since_updated"] = (NOW - proj_df["proj_updated"]).dt.days.fillna(0)
    proj_df["task_completion_ratio"] = proj_df.apply(
        lambda r: r["completed_tasks"] / r["total_tasks"] if r["total_tasks"] > 0 else 0, axis=1)
    proj_df["remaining_days"] = proj_df["planned_end"].apply(
        lambda d: max(0, (d - NOW).days) if pd.notnull(d) else 0)
    return df, proj_df

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        try:
            return joblib.load("model.pkl")
        except Exception as e:
            # Fallback: train on the fly if pickle loading fails (e.g., version mismatch on Streamlit Cloud)
            st.warning(f"Failed to load model.pkl ({e}). Training a new model on the fly...")
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                FEATURES = ['project_age_days', 'days_since_updated', 'completion_pct',
                            'task_completion_ratio', 'total_tasks', 'completed_tasks']
                TARGET   = 'remaining_days'
                
                # Retrieve the cached proj_df
                _, p_df = load_data()
                valid = p_df[FEATURES + [TARGET]].dropna()
                X, y  = valid[FEATURES], valid[TARGET]
                
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                return {
                    'model': model,
                    'features': FEATURES
                }
            except Exception as train_err:
                st.error(f"Failed to train model on the fly: {train_err}")
                return None
    return None

users_df, proj_df = load_data()
model_data = load_model()

# Compute stats
total_u = len(users_df)
total_p = len(proj_df)
verified = int(users_df["isVerified"].sum())
active_p = len(proj_df[~proj_df["is_deleted"]])


# ─── AI Query Engine ──────────────────────────────────────────────────────────
def call_claude(system_prompt: str, user_message: str, max_tokens: int = 800) -> str:
    """Call Anthropic API. Returns response text."""
    import urllib.request
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"].strip()


def build_context_summary() -> str:
    total_users    = len(users_df)
    total_projects = len(proj_df)
    verified       = int(users_df["isVerified"].sum())
    deleted_users  = int(users_df["isDeleted"].sum())
    deleted_proj   = int(proj_df["is_deleted"].sum())
    active_proj    = total_projects - deleted_proj
    avg_pp         = round(total_projects / total_users, 2)

    # Top 5 users by project count
    top_users = (proj_df.groupby("username")["project_id"]
                 .count().sort_values(ascending=False).head(5))
    top_str = "; ".join([f"{u}:{c}" for u, c in top_users.items()])
    usernames_sample = ", ".join(users_df["username"].head(20).tolist())

    return f"""
DATASET SUMMARY:
- Total users: {total_users}
- Total projects: {total_projects} ({active_proj} active, {deleted_proj} deleted)
- Verified users: {verified} | Deleted users: {deleted_users}
- Avg projects/user: {avg_pp}
- Top users by project count: {top_str}
- Sample usernames (first 20): {usernames_sample}
"""


SYSTEM_PROMPT = """You are an intelligent Project Analytics Chatbot. You answer questions ONLY using the dataset context provided.

RULES:
1. Use ONLY the data provided. Never fabricate records or numbers.
2. If data is unavailable, say: "The requested information is not available in the dataset."
3. NEVER reveal passwords, hashes, tokens, or sensitive personal info.
4. For password requests say: "Access to sensitive information is restricted."
5. Be concise and professional.
6. For counts/lists, always give exact numbers from the context.
7. If you need to do arithmetic, show the calculation.

OUTPUT FORMAT:
- Use markdown for tables when listing multiple items.
- Bold key numbers.
- For predictions, explain the basis briefly.
"""


# ─── Intent Classifier + Query Executor ──────────────────────────────────────
class QueryEngine:
    def __init__(self, users_df, proj_df, model_data):
        self.users = users_df
        self.proj  = proj_df
        self.model = model_data

    def _find_user(self, name: str) -> pd.DataFrame:
        name_lower = name.lower().strip()
        mask = (
            self.users["username"].str.lower().str.contains(name_lower, na=False) |
            self.users["email"].str.lower().str.contains(name_lower, na=False)
        )
        return self.users[mask]

    def _find_projects_by_user(self, name: str) -> pd.DataFrame:
        name_lower = name.lower().strip()
        return self.proj[
            self.proj["username"].str.lower().str.contains(name_lower, na=False)
        ]

    def _predict_remaining(self, project_id: int) -> str:
        if self.model is None:
            return "Model not loaded."
        row = self.proj[self.proj["project_id"] == project_id]
        if row.empty:
            return f"Project {project_id} not found in dataset."
        r = row.iloc[0]
        feats = ["project_age_days","days_since_updated","avg_progress",
                 "task_completion_ratio","total_tasks","completed_tasks"]
        X = pd.DataFrame([{
            "project_age_days":      r["project_age_days"],
            "days_since_updated":    r["days_since_updated"],
            "completion_pct":        r["avg_progress"],
            "task_completion_ratio": r["task_completion_ratio"],
            "total_tasks":           r["total_tasks"],
            "completed_tasks":       r["completed_tasks"],
        }])
        pred = max(0, round(self.model["model"].predict(X)[0], 1))
        pname = r["project_name"]
        prog  = round(r["avg_progress"], 1)
        return (f"**Project {project_id}** — *{pname}*\n\n"
                f"- Current avg progress: **{prog}%**\n"
                f"- Predicted remaining: **~{pred} days**\n"
                f"{'🟢 On track' if pred <= 7 else '🟡 Watch closely' if pred <= 20 else '🔴 At risk of delay'}")

    def _detect_delayed(self) -> str:
        now = pd.Timestamp("2026-11-15")
        at_risk = self.proj[
            (self.proj["remaining_days"] == 0) &
            (self.proj["avg_progress"] < 100) &
            (~self.proj["is_deleted"])
        ][["project_id","project_name","username","avg_progress","planned_end"]].head(15)
        if at_risk.empty:
            return "No delayed projects detected."
        tbl = at_risk.rename(columns={
            "project_id": "ID","project_name":"Project","username":"User",
            "avg_progress":"Progress%","planned_end":"Planned End"
        })
        return f"**{len(at_risk)} projects likely delayed:**\n\n" + df_to_markdown(tbl)

    def execute(self, query: str) -> str:
        q = query.lower()

        # ── Security filter
        if any(w in q for w in ["password","hash","token","credential","secret"]):
            return "🔒 Access to sensitive information is restricted."

        # ── Predict remaining days (unified username/user ID/project name/project ID)
        if "remaining" in q or "predict" in q or "days left" in q:
            matched_username = None
            
            # Check for username in query
            for username in self.users["username"].dropna().unique():
                if username.lower() in q:
                    matched_username = username
                    break
                    
            # Check for user ID in query
            if not matched_username:
                m_uid = re.search(r"(?:user|id|user_id)\s*(?:no|num|number)?\s*(\d+)", q)
                if m_uid:
                    uid = int(m_uid.group(1))
                    user_row = self.users[self.users["id"] == uid]
                    if not user_row.empty:
                        matched_username = user_row.iloc[0]["username"]
            
            if matched_username:
                uprojs = self.proj[(self.proj["username"] == matched_username) & (~self.proj["is_deleted"])]
                if uprojs.empty:
                    return f"No active projects found for user **{matched_username}** to predict remaining days."
                
                res = f"🔮 **Project Completion Predictions for {matched_username}**\n\n"
                for _, r in uprojs.iterrows():
                    pid = r["project_id"]
                    pname = r["project_name"]
                    prog = round(r["avg_progress"], 1)
                    
                    if self.model is None:
                        pred_str = "Model not loaded."
                    else:
                        X_feat = pd.DataFrame([{
                            "project_age_days":      r["project_age_days"],
                            "days_since_updated":    r["days_since_updated"],
                            "completion_pct":        r["avg_progress"],
                            "task_completion_ratio": r["task_completion_ratio"],
                            "total_tasks":           r["total_tasks"],
                            "completed_tasks":       r["completed_tasks"],
                        }])
                        pred = max(0, round(self.model["model"].predict(X_feat)[0], 1))
                        status = '🟢 On track' if pred <= 7 else '🟡 Watch closely' if pred <= 20 else '🔴 At risk of delay'
                        pred_str = f"**~{pred} days** ({status})"
                        
                    res += (f"- **Project {pid}** — *{pname}*\n"
                            f"  - Progress: **{prog}%**\n"
                            f"  - Predicted Remaining: {pred_str}\n\n")
                return res

            # If not username, check for project ID/name
            m_pid = re.search(r"(?:project)\s*(\d+)", q)
            pid = None
            if m_pid:
                pid = int(m_pid.group(1))
            else:
                for idx, r in self.proj.iterrows():
                    if r["project_name"].lower() in q:
                        pid = r["project_id"]
                        break
            if pid is not None:
                return self._predict_remaining(pid)

        # ── Total projects count
        if re.search(r"total.*project|number of.*project|projects.*count|how many.*project", q) and not re.search(r"(?:for|of|by|does|has)\s+", q):
            if "active" in q:
                return f"🟢 **Total Active Projects**: **{len(self.proj[~self.proj['is_deleted']])}**"
            if "deleted" in q:
                return f"❌ **Total Deleted Projects**: **{len(self.proj[self.proj['is_deleted']])}**"
            return (f"📁 **Total Projects in Dataset**: **{len(self.proj)}**\n"
                    f"- Active: **{len(self.proj[~self.proj['is_deleted']])}**\n"
                    f"- Deleted: **{len(self.proj[self.proj['is_deleted']])}**")

        # ── Total users count
        if re.search(r"how many.*user|total.*user|user.*count|number of.*user", q):
            if "verified" in q:
                return f"✅ **Total Verified Users**: **{int(self.users['isVerified'].sum())}**"
            if "deleted" in q:
                return f"❌ **Total Deleted Users**: **{int(self.users['isDeleted'].sum())}**"
            return (f"**Total registered users: {len(self.users)}**\n"
                    f"- Verified: {int(self.users['isVerified'].sum())}\n"
                    f"- Deleted accounts: {int(self.users['isDeleted'].sum())}\n"
                    f"- Active accounts: {len(self.users) - int(self.users['isDeleted'].sum())}")

        # ── Unified User Field lookup
        matched_user = None
        for username in self.users["username"].dropna().unique():
            if username.lower() in q:
                matched_user = username
                break
        if not matched_user:
            m_uid = re.search(r"(?:user|id|user_id)\s*(?:no|num|number)?\s*(\d+)", q)
            if m_uid:
                uid = int(m_uid.group(1))
                user_row = self.users[self.users["id"] == uid]
                if not user_row.empty:
                    matched_user = user_row.iloc[0]["username"]

        if matched_user:
            u = self.users[self.users["username"] == matched_user].iloc[0]
            
            # Check for email
            if any(w in q for w in ["email", "mail"]):
                return f"📧 The email address for **{u['username']}** is **{u['email']}**."
                
            # Check for DOB
            if any(w in q for w in ["dob", "date of birth", "birthday", "birth date", "born"]):
                return f"📅 The date of birth (DOB) for **{u['username']}** is **{u['dob']}**."
                
            # Check for verified
            if any(w in q for w in ["verified", "verification"]):
                status = "verified ✅" if u['isVerified'] else "not verified ❌"
                return f"👤 User **{u['username']}** is **{status}**."
                
            # Check for deleted/status
            if "delete" in q:
                status = "deleted ❌" if u['isDeleted'] else "active and not deleted 🟢"
                return f"👤 The account status of user **{u['username']}** is **{status}**."
            if "active" in q:
                status = "active 🟢" if not u['isDeleted'] else "inactive (deleted) ❌"
                return f"👤 User **{u['username']}** is **{status}**."
                
            # Check for profile URL/avatar/image
            if any(w in q for w in ["profile url", "profileurl", "profile_url", "avatar", "image", "url", "link"]):
                return f"🔗 The profile image URL for **{u['username']}** is: {u['profileUrl']}"
                
            # Check for created date/joined/registered
            if any(w in q for w in ["created", "registered", "joined", "member since"]):
                return f"📅 User **{u['username']}** registered / was created on **{u['createdAt']}**."
                
            # Check for updated date
            if "updated" in q:
                return f"📅 User **{u['username']}** profile was last updated on **{u['updatedAt']}**."
                
            # Check for ID
            if "id" in q:
                return f"🆔 The user ID for **{u['username']}** is **{u['id']}**."

        # ── Unified Project Field lookup
        matched_proj_id = None
        m_pid = re.search(r"(?:project)\s*(\d+)", q)
        if m_pid:
            matched_proj_id = int(m_pid.group(1))
        else:
            # Check if any specific project name is in query
            for idx, r in self.proj.iterrows():
                if r["project_name"].lower() in q:
                    matched_proj_id = r["project_id"]
                    break

        if matched_proj_id is not None:
            row = self.proj[self.proj["project_id"] == matched_proj_id]
            if not row.empty:
                r = row.iloc[0]
                
                # Check for progress
                if any(w in q for w in ["progress", "completion"]):
                    return f"📈 The average progress for project **{r['project_name']}** (ID {matched_proj_id}) is **{round(r['avg_progress'], 1)}%**."
                    
                # Check for owner/user
                if any(w in q for w in ["owner", "user", "creator"]):
                    return f"👤 The owner of project **{r['project_name']}** (ID {matched_proj_id}) is **{r['username']}**."
                    
                # Check for status/deleted
                if "delete" in q:
                    status = "deleted ❌" if r['is_deleted'] else "active 🟢"
                    return f"📁 The status of project **{r['project_name']}** (ID {matched_proj_id}) is **{status}**."
                
                # Check for tasks count
                if "task" in q:
                    return f"📋 Project **{r['project_name']}** (ID {matched_proj_id}) has **{r['total_tasks']}** total tasks, with **{r['completed_tasks']}** completed."
                    
                # Check for planned end date/deadline
                if any(w in q for w in ["planned end", "deadline", "end date", "planned_end"]):
                    return f"📅 The planned end date for project **{r['project_name']}** (ID {matched_proj_id}) is **{r['planned_end']}**."
                    
                # Check for created date
                if any(w in q for w in ["created", "creation", "start"]):
                    return f"📅 Project **{r['project_name']}** (ID {matched_proj_id}) was created on **{r['proj_created']}**."

        # ── Project count for specific user
        m = re.search(r"how many projects.+?(does|has|for)\s+(.+?)(?:\s+have|\?|$)", q)
        if not m:
            m = re.search(r"projects.*(of|for|by)\s+(.+?)(\?|$)", q)
        if m:
            name = m.group(2).strip().rstrip("?").strip()
            uprojs = self._find_projects_by_user(name)
            if uprojs.empty:
                return f"No projects found for user matching **'{name}'**."
            active = uprojs[~uprojs["is_deleted"]]
            return (f"**{name.title()}** has **{len(uprojs)}** total project(s) "
                    f"(**{len(active)} active**, {len(uprojs)-len(active)} deleted).")

        # ── List projects for user
        if re.search(r"list.+projects.+of|show.+projects.+of|projects.+list", q):
            m2 = re.search(r"(?:of|for|by)\s+(.+?)(\?|$)", q)
            if m2:
                name = m2.group(1).strip()
                uprojs = self._find_projects_by_user(name)
                if uprojs.empty:
                    return f"No projects found for **'{name}'**."
                tbl = uprojs[["project_id","project_name","avg_progress","is_deleted"]].copy()
                tbl.columns = ["ID","Project Name","Progress%","Deleted"]
                return f"Projects for **{name.title()}** ({len(tbl)} total):\n\n" + df_to_markdown(tbl)

        # ── User details
        if re.search(r"user detail|show.+user|detail.+user|info.+user|about user", q):
            m2 = re.search(r"(?:of|for|about)\s+(.+?)(\?|$)", q)
            if m2:
                name = m2.group(1).strip()
                found = self._find_user(name)
                if found.empty:
                    return f"No user found matching **'{name}'**."
                u = found.iloc[0]
                return (f"👤 **User Details — {u['username']}**\n\n"
                        f"- Verified: {'✅ Yes' if u['isVerified'] else '❌ No'}\n"
                        f"- Account Deleted: {'Yes' if u['isDeleted'] else 'No'}\n"
                        f"- Member Since: {u['createdAt']}")

        # ── ID lookup (general project or user)
        m = re.search(r"(?:details of|show|info|about)?\s*(?:id|user|project)\s*(?:no|num|number)?\s*(\d+)", q)
        if m:
            val = int(m.group(1))
            proj_row = self.proj[self.proj["project_id"] == val]
            user_row = self.users[self.users["id"] == val]
            
            res = ""
            if not proj_row.empty:
                r = proj_row.iloc[0]
                res += (f"📁 **Project Details (ID {val})**\n\n"
                        f"- Name: **{r['project_name']}**\n"
                        f"- Owner: **{r['username']}**\n"
                        f"- Status: **{'Deleted' if r['is_deleted'] else 'Active'}**\n"
                        f"- Progress: **{round(r['avg_progress'], 1)}%**\n"
                        f"- Total Tasks: **{r['total_tasks']}** | Completed: **{r['completed_tasks']}**\n"
                        f"- Planned End: **{r['planned_end']}**\n\n")
            if not user_row.empty:
                u = user_row.iloc[0]
                res += (f"👤 **User Details (ID {val})**\n\n"
                        f"- Username: **{u['username']}**\n"
                        f"- Verified: **{'Yes' if u['isVerified'] else 'No'}**\n"
                        f"- Account Status: **{'Deleted' if u['isDeleted'] else 'Active'}**\n"
                        f"- Created At: **{u['createdAt']}**")
            if res:
                return res
            else:
                return f"No project or user found with ID **{val}**."

        # ── Most projects user
        if re.search(r"highest|most projects|maximum projects|top user", q):
            top = (self.proj.groupby("username")["project_id"]
                   .count().sort_values(ascending=False).head(5))
            lines = "\n".join([f"{i+1}. **{u}** — {c} projects"
                               for i, (u, c) in enumerate(top.items())])
            return f"**Top Users by Project Count:**\n\n{lines}"

        # ── Verified users
        if re.search(r"verified user|all verified", q):
            vu = self.users[self.users["isVerified"] == True][
                ["username","email","createdAt"]].head(20)
            return f"**Verified Users** ({len(self.users[self.users['isVerified']])}):\n\n" + df_to_markdown(vu)

        # ── Active projects
        if re.search(r"active project|show active|list active", q):
            active = self.proj[~self.proj["is_deleted"]][
                ["project_id","project_name","username","avg_progress"]].head(20)
            return (f"**Active Projects** ({len(self.proj[~self.proj['is_deleted']])}):\n\n"
                    + df_to_markdown(active))

        # ── Project summary / stats
        if re.search(r"summary|statistics|overview|stats|dashboard", q):
            total_u  = len(self.users)
            total_p  = len(self.proj)
            verified = int(self.users["isVerified"].sum())
            del_u    = int(self.users["isDeleted"].sum())
            del_p    = int(self.proj["is_deleted"].sum())
            avg_pp   = round(total_p / total_u, 2)
            avg_prog = round(self.proj["avg_progress"].mean(), 1)
            return (
                f"## 📊 Project Summary\n\n"
                f"| Metric | Value |\n|---|---|\n"
                f"| Total Users | **{total_u}** |\n"
                f"| Total Projects | **{total_p}** |\n"
                f"| Active Projects | **{total_p - del_p}** |\n"
                f"| Deleted Projects | **{del_p}** |\n"
                f"| Verified Users | **{verified}** |\n"
                f"| Deleted Users | **{del_u}** |\n"
                f"| Avg Projects/User | **{avg_pp}** |\n"
                f"| Avg Task Progress | **{avg_prog}%** |"
            )

        # ── Delayed / at-risk projects
        if re.search(r"delay|at risk|overdue|behind schedule|likely to be delay", q):
            return self._detect_delayed()

        # ── Search project by name/id (specific ID search fallback)
        m = re.search(r"project\s+(\d+)", q)
        if m:
            pid = int(m.group(1))
            row = self.proj[self.proj["project_id"] == pid]
            if row.empty:
                return f"Project **{pid}** not found."
            r = row.iloc[0]
            return (f"**Project {pid} — {r['project_name']}**\n\n"
                    f"- Owner: {r['username']}\n"
                    f"- Status: {'Deleted' if r['is_deleted'] else 'Active'}\n"
                    f"- Total Tasks: {r['total_tasks']} | Completed: {r['completed_tasks']}\n"
                    f"- Avg Progress: {round(r['avg_progress'],1)}%\n"
                    f"- Planned End: {r['planned_end']}")

        # ── Help
        if re.search(r"help|what can you|capabilities|examples", q):
            return """**I can answer questions like:**

- *How many projects does [name] have?*
- *List all projects of [name]*
- *Show user details of [name]*
- *Which user has the highest number of projects?*
- *Show all verified users*
- *How many users are registered?*
- *Show active projects*
- *Give project summary*
- *Predict remaining days for Project [ID]*
- *Which projects are likely to be delayed?*
- *Show project [ID]*"""

        # ── Direct username lookup (returns count and list of projects)
        for username in self.users["username"].dropna().unique():
            if username.lower() in q:
                u = self.users[self.users["username"] == username].iloc[0]
                uprojs = self.proj[self.proj["username"] == username]
                active = uprojs[~uprojs["is_deleted"]]
                
                res = (f"👤 **User: {u['username']}**\n\n"
                       f"Total projects: **{len(uprojs)}** (**{len(active)} active**, {len(uprojs)-len(active)} deleted).\n\n")
                
                if not uprojs.empty:
                    tbl = uprojs[["project_id","project_name","avg_progress","is_deleted"]].copy()
                    tbl.columns = ["ID","Project Name","Progress%","Deleted"]
                    res += f"**Projects List:**\n\n" + df_to_markdown(tbl)
                else:
                    res += "*No projects found for this user.*"
                return res

        # ── Fallback to Claude AI
        ctx = build_context_summary()
        
        # Extract matching users and projects dynamically from query
        mentioned_users = []
        for username in self.users["username"].dropna().unique():
            if username.lower() in q:
                mentioned_users.append(username)
                
        mentioned_user_ids = []
        for uid in self.users["id"].dropna().unique():
            uid_str = str(uid)
            if f"user {uid_str}" in q or f"id {uid_str}" in q or f"user_id {uid_str}" in q:
                mentioned_user_ids.append(uid)
                
        mentioned_proj_ids = []
        for pid in self.proj["project_id"].dropna().unique():
            pid_str = str(pid)
            if f"project {pid_str}" in q or f"project_id {pid_str}" in q or re.search(rf"\b{pid_str}\b", q):
                mentioned_proj_ids.append(pid)

        # Retrieve relevant user rows
        user_matches = self.users[
            self.users["username"].isin(mentioned_users) | 
            self.users["id"].isin(mentioned_user_ids)
        ]
        if not user_matches.empty:
            ctx += "\n\nRELEVANT USER DETAILS:\n"
            for _, u in user_matches.head(5).iterrows():
                ctx += (f"- Username: {u['username']} | ID: {u['id']} | Email: {u['email']} | "
                        f"DOB: {u['dob']} | Verified: {u['isVerified']} | Deleted: {u['isDeleted']}\n")
                        
        # Retrieve relevant project rows
        proj_matches = self.proj[
            self.proj["username"].isin(mentioned_users) |
            self.proj["user_id"].isin(mentioned_user_ids) |
            self.proj["project_id"].isin(mentioned_proj_ids)
        ]
        if not proj_matches.empty:
            ctx += "\n\nRELEVANT PROJECT DETAILS:\n"
            cols = ["project_id", "project_name", "username", "total_tasks", "completed_tasks", "avg_progress", "is_deleted", "remaining_days"]
            ctx += proj_matches[cols].head(10).to_string(index=False)

        full_prompt = f"{ctx}\n\nUser question: {query}"
        try:
            return call_claude(SYSTEM_PROMPT, full_prompt)
        except Exception as e:
            # Graceful offline/unauthorized fallback
            fallback_res = "⚠️ **AI Service Offline (HTTP 401 Unauthorized)**\n\n"
            fallback_res += "Here is the matching information extracted from your dataset:\n\n"
            
            if "RELEVANT USER DETAILS:" in ctx or "RELEVANT PROJECT DETAILS:" in ctx:
                parts = ctx.split("DATASET SUMMARY:")
                if len(parts) > 1:
                    fallback_res += parts[1].strip()
                else:
                    fallback_res += ctx.strip()
            else:
                fallback_res += build_context_summary().strip()
                
            fallback_res += "\n\n*Configure ANTHROPIC_API_KEY in your environment to enable full AI reasoning.*"
            return fallback_res


engine = QueryEngine(users_df, proj_df, model_data)


# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# ─── Main Layout ─────────────────────────────────────────────────────────────
st.markdown("# 🤖 Project Intelligence Chatbot")
st.markdown("Ask anything about users, projects, or predictions — powered by AI + your dataset.")
st.divider()



# Example questions expander
with st.expander("💡 Example Questions"):
    cols = st.columns(4)
    examples = [
        "Give project summary",
        "Show all verified users",
        "How many users are registered?",
        "Which user has the most projects?",
        "Show active projects",
        "Predict remaining days for Project 101",
        "Which projects are likely to be delayed?",
    ]
    for idx, ex in enumerate(examples):
        col_idx = idx % 4
        with cols[col_idx]:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.user_input = ex
                st.rerun()

# ── Chat Display ──────────────────────────────────────────────────────────────
chat_html = '<div class="chat-container">'
if not st.session_state.messages:
    chat_html += '<div class="bot-msg">👋 Hello! I\'m your Project Intelligence assistant. Ask me about users, projects, stats, or predictions. Type <b>help</b> for examples.</div>'
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += f'<div class="user-msg">🧑 {msg["content"]}</div>'
    else:
        chat_html += f'<div class="bot-msg">🤖 {msg["content"]}</div>'
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────────────────────────────────────
col_in, col_btn, col_clear = st.columns([4, 1, 1])
with col_in:
    user_text = st.text_input(
        "Message",
        value=st.session_state.user_input,
        placeholder="Ask about users, projects, stats or predictions...",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send = st.button("Send ➤", use_container_width=True, type="primary")
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Process Query ─────────────────────────────────────────────────────────────
if (send or st.session_state.get("enter_pressed")) and user_text.strip():
    q = user_text.strip()
    st.session_state.messages.append({"role": "user", "content": q})
    with st.spinner("Thinking..."):
        answer = engine.execute(q)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.user_input = ""
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>🔒 Sensitive data is protected &nbsp;|&nbsp; "
    "🤖 Powered by Claude AI + Scikit-learn &nbsp;|&nbsp; "
    "📊 Data-grounded answers only</small></center>",
    unsafe_allow_html=True,
)
