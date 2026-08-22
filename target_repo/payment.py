def process_payment(user, amount):
    if amount <= 0:
        return False

    print(f"Processing payment of {amount} for {user}")
    return True