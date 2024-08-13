from runai.client import Client

if __name__ == "__main__":
    client = Client()
    # client.do_greeting()
    # #client.swap_speaker()
    # print(client.history)
    # client.do_prompt("What is the meaning of life?")
    # print(client.history)
    # client.do_prompt("How can I find true happiness?")
    # print(client.history)
    # client.do_prompt("What were my first two questions?")
    # print(client.history)
    #
    # # client.do_response()
    # # client.swap_speaker()
    # # print(client.history)
    # #
    # # client.do_response()
    # # client.swap_speaker()
    # # print(client.history)
    while True:
        prompt = input("Enter a prompt: ")
        client.do_prompt(prompt)
        print(client.history[-1]["message"])
