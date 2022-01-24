class Dog:
    "This is a dog class"
    age = 5

    def get_color(self):
        print("grey")


def run_dog():
    print("Executes dog class.")
    # Creates a dog class
    bull_dog = Dog()
    print(bull_dog.age)
    print(bull_dog.get_color)
    print(bull_dog.__doc__)


class ComplexNumber:
    def __init__(self, r=0, i=0):  # constructor
        self.real = r
        self.imag = i

    def get_data(self):
        print(f"{self.real}+{self.imag}j")


def run_cn():
    print("Executes Complex Number class.")
    # Creates a new object
    num1 = ComplexNumber(2, 3)
    # Calls method
    num1.get_data()


if __name__ == "__main__":
    run_dog()
    run_cn()
