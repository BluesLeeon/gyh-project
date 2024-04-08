import pandas as pd
import os
import shutil
from shutil import copyfile
from utils.allcsv_feature_platform import extract_feature_from_csv
from utils.allcsv_feature_platform import get_chinese


def file_to_feature(filepath):
    temp = '\\'.join(filepath.split('\\')[:-1]) + '\\temp'

    if not os.path.exists(temp):
        os.mkdir(temp)

    if filepath.endswith(".xls") or filepath.endswith(".xlsx"):  # 指定要删除的格式，这里是jpg 可以换成其他格式

        name = str(filepath).split('\\')[-1]

        name_chinese = get_chinese(name)

        feature_dict = {
            'name_chineseall': name_chinese
        }

        try:
            exall = pd.read_excel(filepath, sheet_name=None)
            names = list(exall.keys())
            lenint = int(len(exall))

            feature_list = []

            for i in range(0, lenint):
                ex = pd.read_excel(filepath, sheet_name=i)
                if len(ex) != 0:
                    csv_name = temp + r'\{}#{}.csv'.format(filepath.split('\\')[-1], names[i])
                    ex.to_csv(csv_name, encoding="utf-8")
                    feature = extract_feature_from_csv(csv_name)
                    feature_list.append(feature)

            if len(feature_list) > 0:
                heading = []
                row_attribute = []
                column_attribute = []
                allcsv_chinese = []

                for data in feature_list:
                    heading.append(data.get('heading', ''))
                    row_attribute.append(data.get('row_attribute', ''))
                    column_attribute.append(data.get('column_attribute', ''))
                    allcsv_chinese.append(data.get('allcsv_chinese', ''))

                feature_dict['heading'] = ''.join(heading)
                feature_dict['row_attribute'] = ''.join(row_attribute)
                feature_dict['column_attribute'] = ''.join(column_attribute)
                feature_dict['allcsv_chinese'] = ''.join(allcsv_chinese)

                shutil.rmtree(temp)
                os.mkdir(temp)

            return feature_dict

        except BaseException:
            print(filepath, '失败')
            return feature_dict

    else:
        return extract_feature_from_csv(filepath)


if __name__ == '__main__':
    path = r'E:\work\中国移动\all_file\(81)法律援助机构信息.xls'
    # path = r'E:\work\中国移动\all_file\海南统计年鉴_2011_3.2011年鉴-人口.xls'
