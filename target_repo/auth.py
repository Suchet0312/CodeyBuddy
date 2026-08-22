def login(username, password):
    if username == "admin" and password == "1234":
        return True

    return False


def logout(username):
    print(f"{username} logged out")