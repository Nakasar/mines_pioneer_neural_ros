def ask(question, answers=['y', 'n']):
    """Ask a question to the user until the input result belongs to the answers list
    question : str
    answers : str list
    returns the chosen answer"""
    choice = input(question)
    while choice not in answers:
        choice = input(question)
    return choice
