def get_signal(rsi, week52_pos, trust):
    if rsi > 70 or week52_pos > 85:
        return "🔴 AVOID"
    elif rsi < 35 and week52_pos < 30:
        return "🟢 BUY"
    elif trust > 65:
        return "🟡 HOLD"
    else:
        return "⚠️ CAUTION"