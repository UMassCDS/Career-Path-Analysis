import sys
import resume_geocode as geo

loc_str_raw = ' '.join(sys.argv[1:])
coded = geo.geocode_loc(loc_str_raw, 1)
print "{} => {}".format(loc_str_raw, coded)
