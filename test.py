import json

if __name__ == '__main__':
    dict = [{'categorize':1}, {'categorize':1}, {'categorize':1}]
    val = json.dumps(dict)
    print(type(val))
    print(val)