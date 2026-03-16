# March Machine Learning Mania - Winning Strategy Analysis

## Overview

After analyzing 10+ successful solutions from recent competitions, this document outlines the common patterns, key strategies, and a recommended approach for building a competitive solution.

---

## Common Elements Across All Winning Solutions

### 1. Baseline: Bookmaker/Vegas Odds (goto_conversion)

**The single most important finding:** Nearly every medal-winning solution uses **betting odds converted to probabilities** as their baseline or primary ingredient.

- The `goto_conversion` package converts betting odds to win probabilities
- This approach alone achieved medal-level performance in 2024
- Multiple gold medal winners (3rd, 4th place) explicitly stated they used this as their foundation
- Vegas/bookmakers have both the resources and financial incentive to set accurate odds

**Key Resource:** https://github.com/gotoConversion/goto_conversion

### 2. Feature Engineering Hierarchy

All ML-based solutions use a similar feature hierarchy:

| Difficulty | Features | AUC Impact |
|------------|----------|------------|
| **Easy** | Tournament seed, seed difference, men/women flag | ~0.807 |
| **Medium** | Season averages (points, FGA, blocks, PF, rebounds), opponent stats | Moderate |
| **Hard** | Elo ratings (calculated from regular season W/L) | +0.02 AUC |
| **Hardest** | GLM-based team quality scores (point differential regression) | ~0.825 AUC |

### 3. Model Architecture

**XGBoost** is the dominant choice with these common parameters:
```python
param = {
    "objective": "reg:squarederror",  # Predict point differential, not binary
    "booster": "gbtree",
    "eta": 0.009-0.01,
    "subsample": 0.6,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 2,
    "min_child_weight": 4,
    "max_depth": 4,
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_bin": 32-38
}
num_rounds = 700-704
```

**Critical insight:** Models predict **point differential** (continuous), then convert to probabilities using spline interpolation - NOT direct binary classification.

### 4. Cross-Validation Strategy

- **Leave-One-Season-Out (LOSO)** cross-validation
- Train on all seasons except one, validate on held-out season
- Average predictions across all LOSO models for final submission
- This captures year-to-year variance in tournament behavior

### 5. Probability Calibration

Converting point differential predictions to probabilities:
```python
from scipy.interpolate import UnivariateSpline

t = 25  # Clip threshold
spline_model = UnivariateSpline(np.clip(pred, -t, t), label, k=5)
probs = np.clip(spline_model(margin_preds), 0.01, 0.99)
```

---

## Winning Competition Strategies

### Strategy 1: Pure Baseline (Conservative)

Simply use goto_conversion betting odds as your submission. This approach:
- Achieved medal-level scores in 2024
- Requires minimal effort
- Low variance, consistent performance

### Strategy 2: Concentrated Bets (High Risk/High Reward)

The actual **winning strategies** in recent years involved:

1. **Start with strong baseline** (goto_conversion or similar)
2. **Make concentrated, decorrelated bets** on specific upsets

**Mathematical Justification:**
```
f(p) = p(1 - p)^2  # Expected return when betting on upset with probability p

Optimal p = 1/3 (33.3%)
```

Teams with ~33% win probability offer optimal risk/reward for leaderboard climbing.

### Strategy 3: Decorrelation Principle

To maximize chances of winning (not just doing well):
- **Don't bet on popular upsets** - if everyone picks the same upset and it hits, you gain nothing relative to the field
- **Bet on one favorite AND one underdog** - decorrelates from both "boost favorites" and "boost underdogs" strategies
- **Position in sparse outcome space** - unique predictions = unique rewards

### Strategy 4: Domain Knowledge Overrides

The actual 1st place 2025 winner used:
- goto_conversion baseline
- **Manual override betting on Florida to win it all**

Key factors they analyzed:
- Analytics-driven coaching (dedicated analytics director)
- Roster continuity (70% minutes from returners vs Duke's 22%)
- Transfer portal value picks (not top recruits, but specific fits)
- Physical preparation and peaking timing
- Conference tournament performance

| Team | Year | % Minutes from Returners |
|------|------|--------------------------|
| Kansas | 2022 | 81% |
| UConn | 2023 | 53% |
| UConn | 2024 | 61% |
| Florida | 2025 | 70% |
| Duke | 2025 | 22% |

---

## Feature Engineering Details

### Box Score Features (Medium)

```python
boxcols = [
    "Score", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
    "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"
]
```

Calculate for each team:
- Season averages of own performance
- Season averages of opponent performance when playing against this team
- Average point differential

### Elo Rating System (Hard)

```python
def update_elo(winner_elo, loser_elo):
    expected_win = 1.0 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    change = 100 * (1 - expected_win)  # k_factor = 100
    return winner_elo + change, loser_elo - change
```

- Start each team at base_elo = 1000 each season
- Update after each game based on outcome
- Final Elo captures strength adjusted for opponent quality

### GLM Quality Score (Hardest)

```python
import statsmodels.api as sm

formula = "PointDiff ~ -1 + T1_TeamID + T2_TeamID"
glm = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
quality = pd.DataFrame(glm.params)
```

- Fits point differential as function of team indicators
- Coefficients represent team "quality" controlling for opponent
- Quality AUC (~0.825) often beats Seed AUC (~0.807)

---

## Recommended Approach

### Phase 1: Build Strong Baseline
1. Obtain goto_conversion probability matrices OR betting odds data
2. Use as primary submission or blend with ML model

### Phase 2: Build ML Model
1. Create feature hierarchy (easy -> hardest)
2. Train XGBoost with LOSO cross-validation
3. Predict point differential, convert to probabilities via spline
4. Validate Brier score per season

### Phase 3: Ensemble/Blend
```python
final_pred = alpha * goto_conversion_pred + (1 - alpha) * ml_pred
```

### Phase 4: Strategic Overrides (Optional High Risk)
1. Identify 1-2 teams with:
   - ~33% implied win probability (optimal risk/reward)
   - Strong qualitative factors (continuity, analytics, form)
   - Decorrelated from popular picks
2. Override their win probabilities to 1.0 for specific rounds

### Phase 5: Manual Confidence Adjustments
Several solutions applied:
```python
# Increase confidence by 10% if pred < 85%
X['Pred'] = X['Pred'].apply(lambda x: x + x * 0.1 if x < 0.85 else x)
```

---

## Key Insights Summary

| Insight | Importance |
|---------|------------|
| Use betting odds as baseline | Critical |
| Predict point differential, not binary | Very High |
| GLM quality > seed for AUC | High |
| LOSO cross-validation | High |
| Concentrated decorrelated bets | High (for winning) |
| Roster continuity matters | Medium-High |
| Manual overrides with domain knowledge | Medium (high variance) |

---

## Data Sources

- **Competition Data:** Kaggle NCAA dataset (seeds, results, box scores)
- **Betting Odds:** goto_conversion probability matrices
- **External Ratings:** KenPom, Massey Ordinals (men only)
- **Domain Analysis:** Basketball analytics books, game watching, expert analysis

---

## Technical Notes

### Overtime Adjustment
```python
adjot = (40 + 5 * df["NumOT"]) / 40  # Normalize stats for overtime
for col in stat_columns:
    df[col] = df[col] / adjot
```

### Data Doubling
Double dataset by swapping team positions to learn symmetric matchups:
```python
df1.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in columns]
df2.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in columns]
full_data = pd.concat([df1, df2])
```

### AutoGluon Alternative
One solution used AutoGluon with "best_quality" preset:
- CatBoost emerged as top performer
- ~72-73% accuracy on men's, ~83% on women's
- Uses win/loss records, weighted games, opponent strength

---

## Files in Examples Folder

| File | Approach | Key Technique |
|------|----------|---------------|
| `goto-conversion-winning-solution.ipynb` | Betting odds baseline | goto_conversion package |
| `vilnius-ncaa.ipynb` | Full ML pipeline | XGBoost + Elo + GLM quality |
| `final-solution-ncaa-2025.ipynb` | ML + manual overrides | Confidence boosting |
| `hoops-i-did-it-again.ipynb` | AutoGluon | CatBoost ensemble |
| `ncaa2025-3th-place-solution.ipynb` | Baseline + multi-team overrides | Strategic concentrated bets |
| `WRITEUP.md` | 1st place approach | Florida override + domain analysis |
| `APPROACH.md` | Medal strategy | Decorrelation theory |
