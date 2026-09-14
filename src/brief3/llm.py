import json
import os
import time

import mlflow
from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletion
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .pipeline import label_encode, split

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

taxonomy = """
    You must classify each customer message into exactly ONE of these categories:

    balance_enquiry:
        Customer wants to know their account or wallet balance.

    failed_transfer:
        Customer reports that a money transfer failed or was unsuccessful.

    tariff_question:
        Customer asks about fees, charges, tariffs, or transaction costs.

    app_technical:
        Customer reports a technical problem with the mobile application.

    pin_reset:
        Customer wants to reset, change, or recover their PIN.

    account_opening:
        Customer wants to open or register a new account.

    agent_complaint:
        Customer complains about the behavior, service, or actions of an agent.

    card_request:
        Customer wants to request, obtain, replace, or get information about a card.

    praise:
        Customer expresses satisfaction, thanks, compliments, or positive feedback.

    fraud_report:
        Customer reports suspected unauthorized, fraudulent, or suspicious
        financial/account activity.
"""

ambiguity_rule = """
    AMBIGUITY RULE:

    Some customer messages may appear to match multiple categories.

    When this happens, choose the category that represents the
    customer's PRIMARY INTENT.

    Never return multiple categories.

    For example:

    Customer message:
    "I was charged a fee because my transfer failed."

    The primary issue is the failed transfer, so:

    {"label": "failed_transfer"}

    Always select exactly ONE of the ten categories.
    Do not invent new categories.
"""

examples = """
    FEW-SHOT EXAMPLES:

    Customer message:
    "bonjour the transfer is pending since yesterday"

    Output:
    {"label": "failed_transfer"}

    Customer message:
    "abeg l'application ne s'ouvre pas pls"

    Output:
    {"label": "app_technical"}

    Customer message:
    "hi envoyez moi mon solde svp asap"

    Output:
    {"label": "balance_enquiry"}

    Customer message:
    "pls reset my pin asap"

    Output:
    {"label": "pin_reset"}

    Customer message:
    "je veux creer un compte pls"

    Output:
    {"label": "account_opening"}

    Customer message:
    "pls what is the commission on transfer asap"

    Output:
    {"label": "tariff_question"}

    Customer message:
    "agent took commission that is not correct urgent"

    Output:
    {"label": "agent_complaint"}

    Customer message:
    "ma carte est expiree"

    Output:
    {"label": "card_request"}

    Customer message:
    "merci pour le bon service urgent"

    Output:
    {"label": "praise"}

    Customer message:
    "some man don tif ma money i did not authorise this transaction"

    Output:
    {"label": "fraud_report"}   
"""

output_instruction = """
    OUTPUT REQUIREMENTS:

    Return ONLY valid JSON.

    in a case where a batch of messages were sent provide a list of json,
    in the same other in wich ws sent in, emphacies on the same order.
    The JSON must have exactly this structure:

    {"label": "category_name"}

    The label MUST be one of:

    balance_enquiry
    failed_transfer
    tariff_question
    app_technical
    pin_reset
    account_opening
    agent_complaint
    card_request
    praise
    fraud_report

    FOR NO REASON SHOULD YOU RETURN 2 CLASSES FOR A SINGLE MESSAGE.
"""

message = "i see transaction i no expect"

system_promt = f"""
    You are a customer-message classification system.

    Your task is to classify the customer's message into exactly ONE
    of the available categories.
    HERE RE SOM RULES TO FOLLOW:
        {taxonomy}

        {examples}

        {ambiguity_rule}

        {output_instruction}
"""


def get_prediction(message: str, model: str) -> ChatCompletion:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_promt},
            {"role": "user", "content": message},
        ],
    )
    return response


x_train, x_test, x_val, y_train, y_test, y_val = split()

x_test_20 = x_test.iloc[:40].copy()
y_test_20 = y_test.iloc[:40].copy()

encoder = label_encode()
encoder.fit(y_train)


# cleane the json files recieved into valid types for label encode and scoring
def clean_label(raw: str) -> str:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload.get("label", "other")
    except Exception:
        pass

    return raw.strip()


# start invoking the model in a loop getting the predictions
model = "openai/gpt-oss-120b"
mlflow.set_experiment("message_classifier")
with mlflow.start_run(run_name="hosted_groq"):
    predictions = []
    start = time.perf_counter()
    input_tokens = 0.0
    output_tokens = 0.0
    total_tokens = 0.0

    mlflow.log_param("model", f"{model}")
    mlflow.log_param("prompt_variation", "few-shot")

    for msg in x_test_20["message_text"]:
        response = get_prediction(message=msg, model=model)
        prediction = response.choices[0].message.content
        predictions.append(prediction)
        print(prediction)

        if response.usage:
            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.completion_tokens
            total_tokens += response.usage.total_tokens

    latency = time.perf_counter() - start
    clean_predictions = [clean_label(p) for p in predictions]
    compare = zip(clean_predictions, predictions)

    labelt = encoder.transform(clean_predictions)

    # using mlflow to log matrices
    metrics = {
        "recall": recall_score(
            y_test_20, clean_predictions, average="macro", zero_division=0
        ),
        "precision": precision_score(
            y_test_20, clean_predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test_20, clean_predictions, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test_20, clean_predictions, average="weighted", zero_division=0
        ),
        "latency": latency,
    }

    mlflow.log_params(
        params={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    )

    mlflow.log_metrics(metrics)

    mlflow.log_text(
        str(confusion_matrix(y_test_20, clean_predictions)), "confusion_matrix.txt"
    )
