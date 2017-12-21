import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.findall('filename')[0].text    
        for obj in root.iter('object'):
            height = 480
            width = 640
            name = obj.find('name').text  
            if(name == 'cars' or name == 'caar'):
            	name = 'car'
            elif(name == 'ayto' or name =='autor' or name == 'autoclave'):
            	name = 'auto'
            elif(name == 'moto' or name == 'motor' or name == 'motorbikes' or name == 'motorbike cart' or name == 'motobike'):
            	name = 'motorbike'
            elif(name == 'vehicle' or name == 'trucck' or name == 'truk'):
            	name = 'truck'
            elif(name == 'truck crane' or name == 'tractor'):
            	name = 'tractor'
            polygon = obj.find('polygon')
            pts = polygon.findall('pt')
            x_min = width
            x_max = 0
            y_min = height
            y_max = 0
            for pt in pts:
                if int(pt.find('x').text) < x_min:
                    x_min = int(pt.find('x').text)
                if int(pt.find('x').text) > x_max:
                    x_max = int(pt.find('x').text)
                if int(pt.find('y').text) < y_min:
                    y_min = int(pt.find('y').text)
                if int(pt.find('y').text) > y_max:
                    y_max = int(pt.find('y').text)
            values = (filename, height, width, name, x_min, y_min, x_max, y_max)
            xml_list.append(values)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for directory in ["Train","Test"]:
        image_path = os.path.join(os.getcwd(), 'Images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')


main()
