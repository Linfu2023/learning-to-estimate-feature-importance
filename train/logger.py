class Logger:
    def __init__(self, exp_name):
        self.file = open('./{}.log'.format(exp_name), 'w')

    def log(self, content, isprint=True):
        if isprint:
            print(content)
        self.file.write(str(content) + '\n')
        self.file.flush()



