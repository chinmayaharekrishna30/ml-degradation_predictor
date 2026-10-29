# Raw data

Place the real dataset here to use it instead of the synthetic generator:

| File | Source |
|---|---|
| `hotel_bookings.csv` | https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand (check its licence before redistributing) |

Reference material for the monitoring design (not required to run the project):

- NannyML data preparation guide: https://docs.nannyml.com/cloud/model-monitoring/data-preparation
- NannyML repository: https://github.com/NannyML/nannyml
- Time-series drift benchmark (optional, different task): https://data.mendeley.com/datasets/c9zkwnh4px/1

Without `hotel_bookings.csv` the pipeline generates synthetic bookings with the same schema
(`python scripts/generate_synthetic_data.py` writes them here for inspection).
