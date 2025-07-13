from gradio_client import Client

client = Client("https://ihashir-cloud.hf.space")

while True:
    prompt = input("Enter prompt ('exit' to quit):\nYou >> ")
    if prompt.lower() == "exit":
        break

    try:
        print("\n\n...waiting for response, plz wait...\n\n")
        result = client.predict(prompt, api_name="/predict")
        print("___________________________________________")
        print("___________________________________________")
        print("Model >>> ", result)
        print("___________________________________________")
        print("___________________________________________")
    except Exception as e:
        print("Error:", e)
        print("An error occurred while processing your request. Please try again.")