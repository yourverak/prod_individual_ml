import great_expectations as ge
import pandas as pd

def validate_data(data_dict: dict) -> dict:
    report = {
        "valid": True,
        "checks_total": 0,
        "checks_passed": 0,
        "checks_failed": 0,
        "failed_checks": []
    }
    
    if "people" not in data_dict or data_dict["people"].empty:
        return report

    df_people = ge.from_pandas(data_dict["people"])
    
    res1 = df_people.expect_column_values_to_be_unique("user_id")
    _add_check(report, "people", "user_id_unique", res1)

    res2 = df_people.expect_column_values_to_not_be_null("age_bucket")
    _add_check(report, "people", "age_not_null", res2)
    
    res3 = df_people.expect_column_values_to_be_in_set("gender_cd", ["M", "F", "U", "Неизвестно", None])
    _add_check(report, "people", "gender_valid", res3)

    res4 = df_people.expect_column_values_to_not_be_null("region")
    _add_check(report, "people", "region_not_null", res4)

    res5 = df_people.expect_column_values_to_be_between("last_activity_day", min_value="2000-01-01", parse_strings_as_datetimes=True)
    _add_check(report, "people", "activity_date_valid", res5)

    if "transaction" in data_dict and not data_dict["transaction"].empty:
        df_tx = ge.from_pandas(data_dict["transaction"])
        res3 = df_tx.expect_column_values_to_be_unique("transaction_id")
        _add_check(report, "transaction", "tx_id_unique", res3)
        
        res4 = df_tx.expect_column_values_to_not_be_null("merchant_id_tx")
        _add_check(report, "transaction", "merchant_not_null", res4)
        
    if "offer" in data_dict and not data_dict["offer"].empty:
        df_offer = ge.from_pandas(data_dict["offer"])

        res5 = df_offer.expect_column_values_to_not_be_null("start_date")
        _add_check(report, "offer", "start_date_not_null", res5)

    return report

def _add_check(report, table, check_name, res):
    report["checks_total"] += 1
    if res["success"]:
        report["checks_passed"] += 1
    else:
        report["checks_failed"] += 1
        report["valid"] = False
        report["failed_checks"].append({
            "table": table,
            "check": check_name,
            "details": f"Failed {res['result'].get('unexpected_count', 'unknown')} rows"
        })