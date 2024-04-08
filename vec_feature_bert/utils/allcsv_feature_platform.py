import pandas as pd
import re


def is_contain_chinese(check_str):
    """
    判断字符串中是否包含中文
    :param check_str: {str} 需要检测的字符串
    :return: {bool} 包含返回True， 不包含返回False
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def get_chinese(s):
    s = re.sub(
        r"[A-Za-z0-9\!\%\[\]\,\.\：\:\(\)\'\“\”\’\"\‘\-\（\）\、\#\_\。\/\\]",
        '',
        s)
    return s


def empty_str(lil_str):
    if lil_str == '':
        lil_str = 'NaN'
    return lil_str


def get_heading_row(filename):
    # print(filename)
    global heading, row_attribute, rowing
    try:
        df = pd.read_csv(filename, encoding='gbk')
    except:
        df = pd.read_csv(filename, encoding='utf8')

    for i in range(0, 10):
        textline = df.iloc[i].values
        # print(textline)

        if is_contain_chinese(str(textline)):
            row_attribute = str(textline)
            break
        else:
            continue
    for i in range(0, 10):

        textrow = ''.join(str(j) for j in list(df.iloc[:, i]))

        if is_contain_chinese(str(textrow)):
            rowing = str(textrow).split()

            break
        else:
            continue
    return row_attribute, rowing


def extract_feature_from_csv(filepath):
    name = str(filepath).split('\\')[-1]

    name_chinese = get_chinese(name)

    json_allcsv = {
        'name_chineseall': name_chinese
    }

    try:

        row_attribute, column_attribute = get_heading_row(filepath)

        row_attribute = ''.join(get_chinese(str(row_attribute)).split())
        column_attribute = ''.join(get_chinese(str(column_attribute)).split())

        # print(heading,row_attribute)
        try:
            df = pd.read_csv(filepath, encoding='gbk')
        except:
            df = pd.read_csv(filepath, encoding='utf8')

        heading = ''

        heading = ''.join(list(df.columns.values))
        heading = ''.join(get_chinese(str(heading)).split())

        allcsv_chinese = ''.join(get_chinese(str(df.values)).split())
        for point in [id, name, name_chinese, heading,
                      row_attribute, column_attribute, allcsv_chinese]:
            empty_str(point)

        json_allcsv['heading'] = heading
        json_allcsv['row_attribute'] = row_attribute
        json_allcsv['column_attribute'] = column_attribute
        json_allcsv['allcsv_chinese'] = allcsv_chinese

        return json_allcsv

    except BaseException:
        # print('跳过')
        return json_allcsv


if __name__ == '__main__':
    print(extract_feature_from_csv(r'E:\work\中国移动\all_file\(7)公司注销登记_0.csv'))
