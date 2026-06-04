def get_trust_scores(rsi_val, macd_val, macd_sig, week52_pos):

    rsi_score = max(0, min(100, 100 - abs(rsi_val - 50) * 2))

    macd_score = 70 if macd_val > macd_sig else 30

    risk_score = 100 - week52_pos

    trust = (rsi_score + macd_score + risk_score) / 3

    return rsi_score, macd_score, risk_score, trust