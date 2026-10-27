# BharatFinAI — V8 Dynamic Risk Engine

## Scope

V8 adds a modular dynamic-volatility and dynamic-VaR layer to the existing Streamlit application. Legacy stock analysis, Quant, Monte Carlo, and Portfolio tabs are kept in place; only the Risk tab integration is redirected to the dedicated UI module.

## Architecture

```text
data/market_data.py
        |
        v
ui/risk_tab.py
        |
        v
core/risk_engine.py
        |
        +--> core/data_quality.py
        |
        +--> core/volatility_models.py
                   +--> EWMA
                   +--> Realized Volatility
                   +--> GARCH(1,1) [arch]
                   +--> Historical VaR
                   +--> Parametric VaR
                   +--> Student-t VaR
                   +--> GARCH VaR
                   +--> Position-sizing scenario
```

## V8 calculations

### EWMA

The variance update is:

`σ²_t = λ σ²_(t-1) + (1-λ) r²_t`

The UI exposes `λ` from 0.80 to 0.99 and displays the actual value used.

### Realized volatility

Rolling standard deviation is reported for 20D and 60D windows, annualized with `sqrt(252)`.

### GARCH(1,1)

The model is fitted with the `arch` package using daily returns expressed in percentage units. The dashboard reports:

- current conditional volatility
- next-period volatility forecast
- omega / alpha / beta
- persistence (`alpha + beta`)
- optimizer convergence flag
- reliability level and warnings

A failed or missing GARCH installation does not crash the application.

### Dynamic VaR

The dashboard compares:

- Historical VaR
- Parametric Normal VaR
- Student-t VaR
- GARCH-Normal VaR when GARCH succeeds

VaR is shown as a return threshold and converted to an estimated capital loss for the selected capital amount.

VaR is **not** described as a guarantee. It is a statistical estimate conditional on model assumptions and historical information.

## Volatility regime

The classification uses configurable annualized-volatility thresholds:

- LOW: below 12%
- NORMAL: 12% to below 25%
- HIGH: 25% to below 40%
- EXTREME: 40% or above

When at least 30 historical volatility observations exist, the model also reports a historical percentile overlay and promotes the regime when the percentile is unusually high/low.

These thresholds are classification rules, not crash forecasts.

## Position-sizing scenario

The research scenario combines three constraints:

1. stop-distance risk
2. annualized volatility target constraint
3. VaR risk constraint

The result is then dampened using:

`min(data_quality_factor, primary_VaR_reliability_factor)`

The output is intentionally labelled as an educational research-sizing scenario and not personalized financial advice.

## Reliability

Model reliability levels are:

- HIGH
- MEDIUM
- LOW
- INSUFFICIENT DATA

The engine refuses to publish the V8 suite when fewer than 60 usable return observations are available. Individual model functions retain their own lower-level data gates.

Reliability is a statement about data/model conditions, not forecast accuracy.

## Data quality

The V8 data-quality layer checks:

- missing values
- duplicate dates
- non-finite values
- abnormal returns
- extreme daily moves
- stale observations
- zero-return share
- timezone/date alignment

Large daily moves are flagged for inspection rather than automatically removed.

Corporate-action verification requires richer price/event data and is therefore explicitly labelled `NOT_VERIFIED_FROM_RETURN_SERIES`.

## Performance

Market history is retrieved through the existing cached `data/market_data.py` functions (`st.cache_data`). The Risk tab calculates the V8 suite only when the user presses **Calculate V8 Risk**, and GARCH can be skipped using the UI checkbox.

The calculated result is stored in Streamlit session state so normal UI reruns do not repeatedly fit the model.

## Dependencies

V8 requires:

- `scipy`
- `arch`

Both are now explicit requirements. Existing dependencies were otherwise left unchanged.

## Testing

`tests/test_risk_engine.py` includes deterministic tests for:

- data-quality gating
- EWMA
- realized volatility
- regime classification
- Historical VaR
- Student-t VaR
- GARCH graceful failure/success contract
- position sizing
- V8 orchestration
- insufficient-data handling

## Known limitations

- GARCH availability depends on the `arch` package being installed in the deployment environment.
- Yahoo Finance data is an external dependency and can fail, return gaps, or be delayed.
- GARCH VaR uses a Normal innovation distribution; Student-t VaR is separately estimated and not a full GARCH-t model.
- Position sizing uses a transparent 10% annualized volatility target assumption and configurable capital/risk limits; these are prototype assumptions, not individualized recommendations.
- V8 does not yet implement the unified Risk Intelligence Score, stress testing, EVT, full portfolio component VaR, or copula/PCA crisis dependency analysis. Those belong to later roadmap stages.