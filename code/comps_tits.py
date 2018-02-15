import sys
from resume_import import xml2resumes, get_company_name, get_title


def parse_comp_tit(resume_xml):
    resume = []
    experience = resume_xml.find('experience')
    if experience is not None:
        for experiencerecord in experience:
            company_name = get_company_name(experiencerecord, False)
            title = get_title(experiencerecord)
            resume.append((company_name, title))
    return resume


def print_top_comptits(resumes, num):
    company__count = {}
    title__count = {}
    comptit__count = {}

    sys.stderr.write("totalling company, title for {} resumes".format(len(resumes)))
    job_count = 0
    for r, resume in enumerate(resumes):
        if r % 1000 == 0:
            sys.stderr.write("\t{}\n".format(r))

        for comp, tit in resume:
            comptit = (comp, tit)
            company__count[comp] = company__count.get(comp, 0) + 1
            title__count[tit] = title__count.get(tit, 0) + 1
            comptit__count[comptit] = comptit__count.get(comptit, 0) + 1
            job_count += 1

    sys.stderr.write("read {} resumes, {} jobs\n".format(len(resumes), job_count))

    print_top_x(company__count, 50)
    print_top_x(title__count, 50)
    print_top_x(comptit__count, 50)




def print_top_x(label__count, num):
    tups = sorted(label__count.items(), key=lambda x: x[1], reverse=True)
    total = sum(label__count.values())
    for i in range(num):
        lab, count = tups[i]
        print "{:2d}. {:50} {:10d} {:.4f}".format(i, lab, count, float(count)/total)


if __name__ == '__main__':
    ins = sys.argv[1:]

    sys.stderr.write(str(ins) + "\n")
    resumes = xml2resumes(ins, parse_comp_tit)
    print_top_comptits(resumes, 20)










