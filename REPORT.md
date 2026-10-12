# Assignment 1 Report

- **Name**: Gonzalo Domínguez
- **Student ID**: 18691
- **Email**: gdominguez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

This is the Online Shoppers Purchasing Intention Dataset (Sakar and Kastro, 2018), available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)). The local file used is `data/shopping.csv`. 

This dataset contains 12,330 rows and 18 columns. 
- Each row s 1 completed browsing session on a website (entered, browsed, left)
- Each column represents a variable (input, 17 + outcome), and these variables describe the different facets of the browsing activity (number of pages visited, time spent on each category, website analytics, visitor characteristics, calendar...)

The dataset also has no missing values.

The target variable is Revenue, which is not a fixed number but rather a binary indicator of whether a session led to a purchase. 15.5% of sessions were purchasing, and 84.5% were non-purchasing, which makes this an imbalanced, binary classification task.

This dataset is thus a very natural choice — an e-commerce business decision with necessary preprocessing, features with skewed distributions, potential concerns for the later sections of the report, and sets of behavioral ans seasonal features. It's also small enough to commit and retrain comfortably. 

It should be noted that each recorded row represents a completed session, even though business decisions for this project are made at session entry. 

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

Take an e-commerce company. 
They have a pretty limited number of customer support agents, so the company wants to identify, at the beginning of a visit, which visitors appear most likely to make a purchase, thus benefitting from immediate assistance.

The predicted target is **not** the additional sales that offering support may lead to, but just the purchase occurrence.

 
- True Positive: a prioritized visitor ended up purchasing. Bears the base support cost.
- False Positive: a prioritized visitor did not end up purchasing. Bears the base support cost.
- False Negative: a non-prioritized visitor ended up purchasing. Bears the missed-opportunity cost.
- True Negative: a non-prioritized visitor did not end up purchasing. Bears no extra cost.

A False Negative thus isn't a lost sale, and a False Positive does not mean the assistance was worthless because of dataset limitations.

**Operating assumptions**
- Assistance budget: the company can assist up to 20% of sessions at most, a figure that is assessed as a monthly proportion during historical validation.
- Business goal to choose the best policy: maximize monthly purchaser recall (while meeting the assistance constraint just outlined)
- Cost per contact: 1 (units)
- FN cost: 5 (units)
    - This 1:5 ratio is not proven and remains hypothetical, but feels fair because of the department's main goal being to deliver actual support.
- Derived cost formula: 1 * (TP + FP) + 5 * FN



**Temporal evaluation**
I used a monthly chronological split since the intended app wants to predict outcomes for later sessions making use of earlier ones — a random split could mix them from different months and be too optimistic. 

I separated the observations into February-October (Training, 7605), November (Validation, 2998) and December (Testing, 1727). 

The historical expanding-window folds predict July-November by using prior observations, while "policy replay" evaluates September with July and August, October with July-September, and November with July-October. 

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

*Revenue* and *Weekend* were converted to integer 0 and 1s. 
Months were mapped to numbers for splitting and calendar transformations.

The final model uses 12 selected input columns
- *StandardScaler* standardizes the 6 numerical columns
- *OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)* transforms the 6 categorical columns into indicators

**Feature availability**
Because of the dataset's row nature (each row measures a completed session) contrasting with this project's ex-ante nature, it had to be clearly studied which features would be available at the beginning of each visit.

| Input group | Fields supplied to the model |
|---|---|
| Original numeric | `SpecialDay`, `Weekend` |
| Original categorical | `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` |
| Cyclical calendar engineering | `MonthSin = sin(2π × MonthNum / 12)`, `MonthCos = cos(2π × MonthNum / 12)` |
| Visitor/calendar interactions | `ReturningWeekend = 1[returning] × Weekend`, `ReturningSpecialDay = 1[returning] × SpecialDay` |
| Visitor/source interaction | `TrafficVisitorSegment = TrafficType + "__" + VisitorType`, treated categorically |

The last 3 groups are hypotheses regarding calendar periodicity and relationships between type of visitor, and session context.

Categorical traffic codes are not continuous quantities nor do their numerical distances have any meaning.

MonthNum is not a final model input, only being created for splitting and to construct MonthSin and MonthCos. 

The following information is thus excluded from the model input list: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`.
- These fields may still appear in retrospective, descriptive dashboard charts, but the predictor does not make use of them.

**Feature ablation output**

*Note: July–November; reported statistics average the five evaluation months; all candidates were compared using a fixed threshold of 0.25*

| Feature specification | Mean monthly recall | Mean ROC-AUC | Mean log loss | Mean monthly scenario cost/session |
|---|---:|---:|---:|---:|
| A: Original entry-time fields | 0.1167 | 0.6108 | 0.5036 | 0.9522 |
| B: A + cyclical calendar | 0.1294 | **0.6133** | 0.5019 | 0.9508 |
| C: B + visitor interactions | 0.1367 | 0.6118 | 0.5007 | **0.9462** |
| D: C + traffic/visitor segment | **0.1426** | 0.6119 | **0.5001** | 0.9483 |

D was chosen for the pipeline, despite not being the "winner" for all evaluation quantities. D is prioritised because my main goal is to identify eventual purchasers, not to minimize the assumed financial cost (which is subject to hypothesis). 

Of course, the improvement in recall isn't extremely notorious and the additional interactions may have produced some more sparse categories.

It should be noted the ablation does not compare separately optimized operating policies, since they are all evaluated against the final specification's fixed threshold.

**Data limitations present during running**
- Validation and Test transformations produce `OneHotEncoder` warnings for categories not found in earlier training folds. 
- The encoded columns include pretty specific traffic-visitor combinations, which may have limited historical basis.


## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

Since the target is binary, I went for a logistic regression, using L2 regularization, the output of which is a predicted probability for "session purchase". All 3 implementations use the same Feb-Oct rows and 92 column matrix

| Aspect | Scikit-learn | Manual PyTorch | `nn.Module` + `torch.optim` |
|---|---|---|---|
| Model | `LogisticRegression` | Logits `X @ weights + bias` | Single `nn.Linear(n_features, 1)` layer |
| Binary loss | Logistic loss | `binary_cross_entropy_with_logits` | `BCEWithLogitsLoss` |
| Optimization | `lbfgs` | Explicit parameter update after autograd | `torch.optim.SGD` |
| L2 penalty | `C = 0.1` | `sum(weights²)/(2 × n × C)` | Same weight penalty |
| Intercept/bias penalty | Not penalized | Not penalized | Not penalized |
| Learning rate | Solver-managed | 1.0 | 1.0 |
| Iterations / epochs | Converged in 84 iterations in the logged run | 5,000 epochs | 5,000 epochs |
| Floating-point representation | Model input converted to float64 | float64 tensors | float64 model/tensors |

The PyTorch models start with 0 weights and bias, while the manual version computes gradients via loss.backward(), applying parameter updates, then clearing gradients. The standard uses optimizer.zero_grad(), loss.backward() and optimizer.step()

The C candidates were .1 and 1, which were selected using mean historical log-loss. .1 was chosen for the final run (July–August selection scores: 0.457 vs 0.471).


**Model agreement**

| Diagnostic | Scikit-learn | Manual PyTorch | Standard PyTorch |
|---|---:|---:|---:|
| February–October training log loss | 0.344358 | 0.344359 | 0.344359 |
| November validation log loss | 0.582616 | 0.582616 | 0.582616 |

Maximum absolute **December probability** difference: scikit-learn vs manual = **0.00013171**; scikit-learn vs standard PyTorch = **0.00013171**; manual vs standard PyTorch = **0.0**.


**December holdout comparison (0.50 classification threshold, "Standard")**

| Model | Accuracy | Precision | Recall | ROC-AUC | Log loss | Scenario cost |
|---|---:|---:|---:|---:|---:|---:|
| Naive constant training rate baseline | 0.8749 | 0 | 0 | 0.5000 | 0.3769 | 1,080 |
| Scikit-learn | 0.8749 | 0 | 0 | 0.6391 | 0.3663 | 1,080 |
| Manual PyTorch | 0.8749 | 0 | 0 | 0.6391 | 0.3663 | 1,080 |
| Standard PyTorch | 0.8749 | 0 | 0 | 0.6391 | 0.3663 | 1,080 |

The learned models have higher ROC-AUC and lower log loss than the constant baselines.


**December holdout, frozen 0.25 threshold (Policy A):**

| Model | Contacts | TP | FP | FN | TN | Precision | Recall | Scenario cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scikit-learn | 136 | 31 | 105 | 185 | 1,406 | 0.2279 | 0.1435 | 1,061 |
| Manual PyTorch | 136 | 31 | 105 | 185 | 1,406 | 0.2279 | 0.1435 | 1,061 |
| Standard PyTorch | 136 | 31 | 105 | 185 | 1,406 | 0.2279 | 0.1435 | 1,061 |
| Always contact nobody (reference) | 0 | 0 | 0 | 216 | 1,511 | 0 | 0 | 1,080 |

This policy contacts 7.87% of December visitors. Purchasers represent 22.79% of these contacts, while December's purchasing rate is at 12.51%. 

This means that, under the hypothetical cost scenario outlined earlier, we are saving 19 (units) in comparison to contacting nobody.

**Historical policy**

I ran 3 fixed-cutoff strategies, replayed on Sep-Nov, using calibration info for each evaluation month.

| Historical replay (Sep–Nov) | Mean monthly recall | Mean monthly contact share | Highest monthly contact share | Within 20% in every replay month? |
|---|---:|---:|---:|---|
| A: Conservative historical capacity cutoff | 0.2951 | 0.2288 | 0.3772 | No |
| B: Historical 80th-percentile cutoff | 0.4815 | 0.3839 | 0.5268 | No |
| C: Historical quantile plus cost-based floor | 0.4388 | 0.3316 | 0.3772 | No |

The 3 of them broke the 20% contact-share limit we stated, and so the fallback kept Policy A. This is in no way evidence that A respects capacity.

Policy A's cutoff is 0.25, using historical folds for Jul-Nov.


## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

- Probability mismatch in November (16.36% vs. actual positive rate of 25.35%). It could help to investigate the calibration across the past months, and trying to see if more recent training data improves the reliability of these probabilities.
- The 20% hypothetical staffing cap is never fully in force, with all candidates exceeding it at some point. Perhaps an adaptive cutoff of sorts could be implemented — if timestamps existed, these could use actual visitor arrival times and the corresponding staff availability.
- Very limited temporal information, since there is nothing deeper than month (week, day, timestamp...). The dataset definitely struggles because of that, so further data collection would be very helpful.
- The cost scenarios are fully hypothetical and may have led to a biased conclusion. If I could obtain the cost of customer assistance, the specific outcome of a proper assistance, or information on customer-value, that would enhance the choice of economic penalty.
- On that same line, since the outcome of actually assisting is unoberved, perhaps running an experiment that could determine if proactive assistance actually changes purchase likelihood or customer satisfaction would amplify the scope of the project.


## Generative AI use disclosure

AI was used for four different goals: code refinement, app development, completionism, and proper functionality.
- Code refinement: more complex and detailed parts of the code were produced by AI, starting off of a baseline created by me.
- App development: the Gradio app is 99% created by AI, as authorised in-class by the instructor — I pulled off a very basic iteration of it by hand but I would have never been able to do it without it.
- Completionism: While evaluating potential improvements, AI came up with a couple good ideas that it helped explain and then implement. This includes the historical policy experiment, and the rolling month validation.
- Proper functionality: identification and understanding of errors in-code and during running, learning how to properly commit, etc.
