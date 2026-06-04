import os
import pandas as pd
from datetime import datetime

FEEDBACK_FILE = "data/feedback.csv"


def save_feedback(name, city, user_type, rating, confusion, suggestion):

    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "city": city,
        "user_type": user_type,
        "rating": rating,
        "confusion": confusion,
        "suggestion": suggestion
    }

    df_new = pd.DataFrame([feedback_data])

    if os.path.exists(FEEDBACK_FILE):

        df_old = pd.read_csv(FEEDBACK_FILE)

        df = pd.concat(
            [df_old, df_new],
            ignore_index=True
        )

    else:

        df = df_new

    df.to_csv(FEEDBACK_FILE, index=False)


def load_feedback():

    if os.path.exists(FEEDBACK_FILE):

        return pd.read_csv(FEEDBACK_FILE)

    return pd.DataFrame() 