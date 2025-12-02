# ---------------------------------------------------------
# test_gvpo_pipelines.py
# 自动测试 _quick_gvpo_regression_test() 和 _full_gvpo_test_pipeline()
# 放在项目根目录即可运行：
#
#   poetry run python test_gvpo_pipelines.py
#
# ---------------------------------------------------------

import traceback

def run_test(fn, fn_name):
    print("\n" + "=" * 70)
    print(f"Running {fn_name}() ...")
    print("=" * 70)

    try:
        fn()
        print(f"\n[OK] {fn_name}() finished successfully.")
    except Exception as e:
        print(f"\n[ERROR] {fn_name}() failed:")
        print("-" * 70)
        traceback.print_exc()
        print("-" * 70)


def main():
    print("\n==============================================")
    print(" GVPO Regression + Full Pipeline Test Launcher ")
    print("==============================================\n")

    # Import inside main() to avoid errors before user patches the code.
    try:
        from pycam.allo_util_RL import (
            _quick_gvpo_regression_test,
            _full_gvpo_test_pipeline,
        )
    except Exception:
        print("[FATAL] Could NOT import GVPO test functions.")
        print("Please make sure you added the two functions in allo_util_RL.py")
        traceback.print_exc()
        return

    # ---------------------------
    # 1) Quick sanity test
    # ---------------------------
    run_test(_quick_gvpo_regression_test, "_quick_gvpo_regression_test")

    # ---------------------------
    # 2) Full pipeline benchmark
    # ---------------------------
    run_test(_full_gvpo_test_pipeline, "_full_gvpo_test_pipeline")

    print("\nAll tests completed.\n")


if __name__ == "__main__":
    main()
