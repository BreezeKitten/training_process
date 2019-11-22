#import file_manger
import sys

def test():
    print('9999')

if __name__ == '__main__':
    #FM = file_manger.file_manger('log_test', '1134')
    temp = sys.stdout
    f = open('aa.txt', 'a+')
    sys.stdout = f
    print('test')
    test()
    sys.stdout = temp
    f.close()

    
    
        