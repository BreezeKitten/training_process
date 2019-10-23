import json

gamma = 0.8

def Read_data(file_name):
    data = {}
    file = open(file_name,'r')
    data_line = file.readline()
    count = 0
    while(data_line):
        data[count] = json.loads(data_line)
        data_line = file.readline()
        count = count + 1
    file.close()
    return data

def Calculate_value(data_without_value):
    time_now = 0
    Start_item = 0
    data_with_value = {}
    for item in data_without_value:
        if data_without_value[item]['time_tag'] < time_now:
            data_without_value[item]['time_tag'] = time_now + 0.1
            for i in range(Start_item,item):
                data_with_value[i] = data_without_value[i]
                data_with_value[i]['Value'] = pow(gamma,(data_without_value[item]['time_tag'] - data_without_value[i]['time_tag'])*10)
            data_with_value[item] = data_without_value[item]
            data_with_value[item]['Value'] = 1
            Start_item = item + 1
            time_now = 0
        else:
            time_now = data_without_value[item]['time_tag'] 
    return data_with_value

def Record_data(data,file_name):
    file = open(file_name,'a+')
    line = '\n'
    for item in data:
        json.dump(data[item],file)
        file.writelines(line)
    file.close()
    

if __name__ == '__main__':
    data_without_value = Read_data('record_vw.json')
    data = Calculate_value(data_without_value)
    Record_data(data,'record.json')