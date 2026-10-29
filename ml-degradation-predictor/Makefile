.PHONY: install pipeline pipeline-real dashboard api test data clean

install:
	pip install -r requirements.txt

data:
	python scripts/generate_synthetic_data.py

pipeline:
	python scripts/run_pipeline.py --source synthetic

pipeline-real:
	python scripts/run_pipeline.py --source real

dashboard:
	streamlit run app/dashboard.py

api:
	uvicorn api.main:app --reload

test:
	pytest -q

clean:
	rm -rf models/*.joblib reports/figures/*.png reports/*.json reports/*.csv reports/RESULTS.md \
	       data/processed/* .pytest_cache
