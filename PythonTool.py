import matplotlib.pyplot as plt  # plt.show(block=False) plt.savefig(path)
import re


def parse_and_execute_python_code(tool, context, sql_results):
    code_match = re.search(r"```python\n([\s\S]*?)```", tool)
    code = code_match.group(1)
    print(f"Parsed Python code: {code}")
    code = code.strip()
    exec_context = {}
    python_res = ""
    if sql_results:
        exec_context["sql_results"] = sql_results
        print(f"Injecting SQL results into Python code execution: {sql_results}")

    try:
        print(f"Executing Python code: {code}")
        print(f"exec_context: {exec_context}")
        exec(code, exec_context)
        context["python_results"] = "Python results obtained"
        python_res = "Python results obtained"
        print("Python results obtained")
    except Exception as e:
        print(f"Error executing Python: {e}")
        context["error"] = f"Python error: {e}"

    return context, python_res
