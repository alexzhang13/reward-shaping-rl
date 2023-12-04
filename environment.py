import utils.query_llm as query_llm

# def reward(a: int):
    # pass

def main():
    code = """def reward(a: float, b: float, c: float, d: float):
\treturn a
    """
    try:
        exec("print(code)")
        exec(code, globals())
        _ = reward(0, 0, 0, 0)
    except:
        print("Error!")
    reward(2, 2, 2, 2)

if __name__ == "__main__":
    main()
