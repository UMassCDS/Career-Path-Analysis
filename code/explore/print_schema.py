import sys
import gzip
import xml.etree.ElementTree as ET


def print_kids(elt, elt__kids, indent=0):
    delimit = '/'
    if delimit in elt:
        prefix, tag = elt.rsplit("/", 1)
        print " " * indent, tag
    else:
        print " " * indent, elt

    if elt in elt__kids:
        for kid in sorted(elt__kids.get(elt, [])):
            print_kids(kid, elt__kids, indent + 4)


###################################

if __name__ == '__main__':

    infile_name = sys.argv[1]
    sys.stderr.write("parsing xml {}\n".format(infile_name))

    elt__children = {}
    schema_todo = {'resume'}
    schema_done = set()

    if infile_name.endswith('.gz'):
        with gzip.open(infile_name, 'rb') as infile:
            tree = ET.parse(infile)
    else:
        tree = ET.parse(infile_name)
    root = tree.getroot()

    while len(schema_todo) > 0:

        elt_tag = schema_todo.pop()
        elt__children[elt_tag] = set()
        sys.stderr.write("exploring {}\n".format(elt_tag))

        for i, elt in enumerate(root.findall(elt_tag)):
            for elt_child in elt:
                elt_child_tag = elt_tag + '/' + elt_child.tag

                if (elt_child_tag not in schema_todo) and (elt_child not in schema_done):
                    schema_todo.add(elt_child_tag)
                if elt_child_tag not in elt__children[elt_tag]:
                    sys.stderr.write("\t found {}\n".format(elt_child_tag))
                    elt__children[elt_tag].add(elt_child_tag)

        schema_done.add(elt_tag)

    print_kids("resume", elt__children)


