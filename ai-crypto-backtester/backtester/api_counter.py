api_call_count = 0

def increment_api_count():
    global api_call_count
    api_call_count += 1

def get_api_call_count():
    return api_call_count
