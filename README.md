# DealMakers — Adaptive Competitive Pricing Agent

A competitive dynamic pricing project combining segmented demand modeling, inventory-aware pricing, and opponent-adaptive strategy selection.

## Overview

The project consists of two parts:

- **Part 1 — Demand Estimation & Price Optimization:** estimate purchase probability and choose revenue-maximizing prices under inventory constraints.
- **Part 2 — Competitive Pricing:** compete against other pricing agents and adapt strategy based on observed opponent behavior.

Our final team solution used an adaptive meta-agent that selected between two pricing strategies after observing the opponent during the first 80 interactions.

## My Contributions

My primary contributions were:

- Built an **eight-segment Logistic Regression demand-modeling pipeline** using median splits on three customer covariates.
- Trained and serialized the segment-specific demand models into `8_models_dict.pkl`.
- Implemented the **DP-based pricing agent** in `david_agent.py`.
- Combined demand estimates, inventory-aware DP pricing, adaptive segment multipliers, and opponent-aware price adjustments.
- Contributed the DP agent to the team's final adaptive meta-agent.

The final `dealmakers.py` submission integrates my DP agent with a teammate's non-DP pricing strategy and is included unchanged for completeness.

## Segment-Specific Demand Modeling

Customers are divided into eight segments using high/low splits on three covariates.

For each segment, a separate Logistic Regression model estimates purchase probability from:

- `Covariate1`
- `Covariate2`
- `Covariate3`
- `Price`

The workflow is implemented in:

```text
notebooks/logistic_8_segments.ipynb
```

The eight trained models are serialized into:

```text
agents/dealmakers/8_models_dict.pkl
```

For static price optimization, candidate prices are evaluated using:

```text
Expected Revenue = Price × Purchase Probability
```

The broader team also explored segmented tree-based models, and the final Part 1 team approach used eight segmented XGBoost models with an inventory-pressure multiplier.

## DP Pricing Agent

My pricing agent is implemented in:

```text
agents/dealmakers/david_agent.py
```

It combines:

- segment-specific Logistic Regression demand models
- a precomputed inventory-aware DP pricing policy
- adaptive segment-level price multipliers
- static expected-revenue price comparison
- recent opponent-price reactions

The DP policy is stored in:

```text
agents/dealmakers/dp_policy.pkl
```

For each customer segment, it maps:

```text
policy[segment][inventory][time_remaining] -> price
```

The policy was computed offline using segment-specific demand estimates and future revenue under inventory constraints.

The original policy-generation notebook is not included in this repository.

## Adaptive Meta-Agent

The final team agent is implemented in:

```text
agents/dealmakers/dealmakers.py
```

It combines:

- `DavidSubAgent` — DP-based pricing strategy
- `NewSubAgent` — XGBoost demand estimation with an inventory- and market-saturation-based pricing strategy

During the first **80 steps**, the agent records opponent prices and measures:

- `price_std`
- `frac_small_move`

The final implementation uses:

```text
frac_small_move > 0.7 and price_std < 15
```

to select the DP sub-agent for a static/predictable opponent behavior profile. Otherwise, it uses the non-DP strategy.

## Strategy Motivation

Local simulations showed that different pricing strategies could interact very differently:

| Matchup | Approximate Revenue |
|---|---:|
| Multiplier vs. Multiplier | ~10K–12K each |
| Multiplier vs. DP | DP ~10K–11K, Multiplier ~7K–8K |
| DP vs. DP | ~3K–4K each |

DP-vs-DP competition often created a price war that reduced revenue for both agents. This motivated using an adaptive strategy selector instead of relying on one pricing policy against every opponent.

## Repository Structure

```text
.
├── agents/
│   └── dealmakers/
│       ├── 8_models_dict.pkl
│       ├── 8_xgb.pkl
│       ├── david_agent.py
│       ├── dealmakers.py
│       └── dp_policy.pkl
├── notebooks/
│   └── logistic_8_segments.ipynb
├── report/
│   └── final_report.pdf
├── requirements.txt
└── README.md
```

## Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- NumPy
- pandas
- scikit-learn
- XGBoost

## Team

- Alice Lee
- Pin-Hsuan (David) Lai
- Pin-Yeh Lai

## Course

**ORIE 5355 — Applied Data Science: Decision-Making Beyond Prediction**
