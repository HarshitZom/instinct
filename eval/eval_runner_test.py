from eval_suite.eval_runner import EvalRunner

if __name__ == "__main__":
    runner = EvalRunner(
        model="instinct",
        dataset_lang="Typescript",
        dataset_version="v1",
    )
    results = runner.run_evals(["devtime"], verbose=True)
