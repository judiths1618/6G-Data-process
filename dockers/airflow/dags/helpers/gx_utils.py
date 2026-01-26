# helpers/gx_utils.py
import great_expectations as gx

def get_gx_context():
    return gx.get_context(
        context_root_dir="/opt/airflow/great_expectations"
    )
