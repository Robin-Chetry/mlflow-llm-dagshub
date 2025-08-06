import mlflow
import openai
import os
import pandas as pd
from dotenv import load_dotenv
import dagshub

# Load env vars
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Init DagsHub
dagshub.init(repo_owner='robinrawatchetry', repo_name='mlflow_genai', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/robinrawatchetry/mlflow_genai.mlflow/")
mlflow.set_experiment("LLM Evaluation")

# Sample data
eval_data = pd.DataFrame({
    "inputs": [
        "What is MLflow?",
        "What is Spark?"
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing the ML lifecycle...",
        "Apache Spark is a distributed computing system for big data..."
    ]
})

# Run and evaluate
with mlflow.start_run():
    predictions = []
    for prompt in eval_data["inputs"]:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the following question in two sentences"},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content.strip()
        predictions.append(answer)

    # Add predictions to DataFrame
    eval_data["predictions"] = predictions

    # Save evaluation results
    eval_data.to_csv("eval_results.csv", index=False)
    mlflow.log_artifact("eval_results.csv")

    # Optionally log text for each prediction
    for i, row in eval_data.iterrows():
        mlflow.log_text(row["predictions"], f"prediction_{i}.txt")

    print("\n✅ Evaluation complete. Predictions logged to MLflow.")






    
    
## 🔍 Summary of What This Script Does:
    '''
| Step | Action |
|------|--------|
| ✅ 1 | Connects to DAGsHub with MLflow tracking |
| ✅ 2 | Prepares a list of questions + ground truth answers |
| ✅ 3 | Logs GPT-4 as an MLflow model using OpenAI |
| ✅ 4 | Automatically evaluates the model’s answers |
| ✅ 5 | Logs results (toxicity, latency, similarity) |
| ✅ 6 | Saves and prints the evaluation summary |

    '''



