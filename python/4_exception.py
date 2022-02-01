name = input("Enter your name: ")
print("Hello there, {}!".format(name.title()))

while True:
    try:
        num = int(input("Enter an integer: "))
    except ValueError as e:
        print('That was not a number.')
        print("E occurred: {}".format(e))
    except KeyboardInterrupt:
        print('Exiting execution...')
        break
    else:
        print("ok, thanks {} for entering {}".format(name.title(), num))
        break
    finally:
        print('>>>')

result = eval(input("Enter an expression: "))
print(result)