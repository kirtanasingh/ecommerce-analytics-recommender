# E-Commerce Analytics and Recommendation System

Big data project using Apache Spark (PySpark) to analyze customer purchasing behavior and recommend products on a real online retail dataset.

## What it does
- Customer segmentation using K-Means on RFM features
- Frequently bought-together products using FP-Growth
- Personalized recommendations using ALS (collaborative filtering)
- Similar-product recommendations using TF-IDF (content-based filtering)
- Evaluation with silhouette score, support/confidence/lift, Precision@K, Recall@K, NDCG@K

## Dataset
**Online Retail II** (UCI Machine Learning Repository)
Link: https://archive.ics.uci.edu/dataset/502/online+retail+ii

Real transactions from a UK online gift-ware retailer, 2009 to 2011 (about 1 million raw rows, CC BY 4.0).
The dataset is not included in this repo. Download it from the link above.

**After cleaning** (removed missing customers, cancellations, and non-positive quantity/price):

| Metric | Value |
|---|---|
| Transaction rows | 805,549 |
| Orders | 36,969 |
| Customers | 5,878 |
| Products | 4,631 |

## Tech used
Python, PySpark 4.0 (Spark MLlib), pandas, scikit-learn, matplotlib, Google Colab

## How to run
1. Open `ecommerce_analytics_recommender.ipynb` in Google Colab
2. Download the dataset from the link above, convert the Excel file to `/content/online_retail.csv`, and upload it to Colab
3. Run all cells in order

## Results

### 1. Customer segmentation (K-Means on log-scaled RFM)

Silhouette scores tried for k = 2 to 8: k=2 gave 0.624, k=3 0.550, k=4 0.556, k=5 0.543. We chose **k = 4** (silhouette **0.556**) because it gives more useful business segments than k=2, with only a small drop in score.

| Segment | Customers | Avg recency (days) | Avg orders | Avg spend (GBP) | Interpretation |
|---|---|---|---|---|---|
| 0 | 1,366 | 56.1 | 17.9 | 9,998.0 | **Champions**: recent, frequent, highest spend |
| 3 | 1,601 | 61.3 | 3.0 | 867.5 | **Recent / promising**: active lately, moderate spend |
| 2 | 1,211 | 309.7 | 4.5 | 1,812.5 | **At-risk**: decent past value, gone quiet |
| 1 | 1,700 | 450.7 | 1.3 | 295.6 | **Lost / one-time buyers**: long inactive, low spend |

### 2. Frequent pattern mining (FP-Growth)

Settings: top 300 products, baskets of 2 to 12 items (22,166 baskets), minSupport = 0.02, minConfidence = 0.3. Result: 2 frequent itemsets of size 2 and 3 rules.

| Rule | Confidence | Lift | Support |
|---|---|---|---|
| WOODEN PICTURE FRAME WHITE FINISH → WOODEN FRAME ANTIQUE WHITE | 0.58 | 13.09 | 0.021 |
| WOODEN FRAME ANTIQUE WHITE → WOODEN PICTURE FRAME WHITE FINISH | 0.48 | 13.09 | 0.021 |
| RED HANGING HEART T-LIGHT HOLDER → WHITE HANGING HEART T-LIGHT HOLDER | 0.67 | 5.28 | 0.024 |

All rules have lift well above 1, meaning these items are bought together far more often than by chance. Business use: bundle or cross-sell wooden frames and the hanging heart T-light holders.

### 3. Collaborative filtering (ALS, implicit feedback, rank = 10)

Purchase quantity was used as implicit feedback (481,932 customer-product interactions, 80/20 train-test split). Evaluated at K = 10:

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity baseline | 0.0244 | 0.0196 | 0.0261 |
| ALS | 0.0204 | 0.0214 | 0.0227 |

ALS MAP@10 = 0.0060.

**Reading these numbers honestly:** all values are low, which is normal for sparse purchase data (4,631 products, most customers buy only a small fraction). ALS has slightly higher recall than the popularity baseline but slightly lower precision and NDCG, so on this dataset the personalized model does not clearly beat "recommend bestsellers". Tuning rank, regParam and alpha, or combining ALS with popularity or content signals, could improve it.

Example output for customer 17865 (Champions segment), top 5 recommendations:
SMALL CHINESE STYLE SCISSOR, ASSORTED WHITE EMBOSSED CHINA MUGS, SMALL CHUNKY GLASS ROMAN BOWL, MIRRORED DISCO BALL, ASSORTED LAQUERED INCENSE HOLDERS.

### 4. Content-based filtering (TF-IDF + cosine similarity)

Built on product descriptions (4,631 products). Example for **LUNCH BAG RED SPOTTY**:

| Similar product | Similarity |
|---|---|
| LUNCH BAG RED VINTAGE DOILY | 0.613 |
| LUNCH BAG CARS BLUE | 0.604 |
| TEA BAG PLATE RED SPOTTY | 0.511 |
| RED SPOTTY CANDY BAG | 0.457 |
| LUNCH BAG WOODLAND | 0.456 |

Content-based filtering works without any purchase history, so it helps with new products (cold start).

## Key insights
- About 23% of customers (the Champions segment) are recent, frequent and spend the most (avg about GBP 10k each), so they are the priority for retention.
- About 29% of customers are lost one-time buyers (avg recency over 450 days), which is a clear win-back target.
- Strongest product combinations are similar items from the same family (picture frames, hanging heart T-light holders).
- On sparse implicit data, a simple popularity baseline is hard to beat; ALS gives better recall but needs tuning.

## Limitations and future work
- No explicit ratings in the dataset, so quantity was used as implicit feedback
- FP-Growth was limited to the top 300 products to avoid memory crashes on free Colab, so rare combinations are not captured
- ALS hyperparameters were not tuned
- Future work: hybrid recommender (ALS + content + popularity), real-time streaming with Kafka, deployment as an API
