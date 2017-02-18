__author__ = 'alex'

import os
import xml.etree.ElementTree as ElementTree

# some data
nodeID_key = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}nodeID'
annotations_tag = '{http://www.abbyy.com/ns/Aux#}TextAnnotations'
label_tag = '{http://www.w3.org/2000/01/rdf-schema#}label'

instance_annotation_tag = '{http://www.abbyy.com/ns/Aux#}InstanceAnnotation'
annotation_start_tag = '{http://www.abbyy.com/ns/Aux#}annotation_start'
annotation_end_tag = '{http://www.abbyy.com/ns/Aux#}annotation_end'
annotation_instance_tag = '{http://www.abbyy.com/ns/Aux#}instance'
document_text_tag = '{http://www.abbyy.com/ns/Aux#}document_text'


# write to orig.txt and norm.txt
def parse_file(file):
    norm_with_node_id = {}
    orig_with_node_id = {}

    root = ElementTree.parse(file).getroot()
    for child in root:
        # если TextAnnotations, обрабатываем отдельно
        if child.tag == annotations_tag:
            document_text = child.find(document_text_tag).text

            for annotation in child.iter(instance_annotation_tag):
                instance = annotation.find(annotation_instance_tag)
                if nodeID_key in instance.attrib:
                    node_id = instance.attrib[nodeID_key]

                    start_pos = int(annotation.find(annotation_start_tag).text)
                    end_pos = int(annotation.find(annotation_end_tag).text)
                    # выделить подстроку из document_text от start до end
                    actual_text = document_text[start_pos:end_pos]
                    orig_with_node_id[node_id] = actual_text
        elif nodeID_key in child.attrib:
            # иначе
            # get node_id
            node_id = child.attrib[nodeID_key]
            # find label
            label = child.find(label_tag)
            if label is not None:
                normalized_text = label.text
                norm_with_node_id[node_id] = normalized_text

    for key, value in orig_with_node_id.items():
        if key in norm_with_node_id:
            print(value, file=orig_file)
            print(norm_with_node_id[key], file=norm_file)


orig_file = open('parsed/orig.txt', 'w')
norm_file = open('parsed/norm.txt', 'w')

for r, subdirs, files in os.walk('data/corpora'):
    for filename in files:
        source_file = os.path.join(r, filename)
        print('parsing %s' % source_file)
        parse_file(source_file)

orig_file.close()
norm_file.close()
