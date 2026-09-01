class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email

    def get_profile(self):
        return {
            "username": self.username,
            "email": self.email
        }