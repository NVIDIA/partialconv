import yaml
import os
f = os.path.join(os.path.split(__file__)[0], r"./real.yaml")
y = yaml.load(open(f), yaml.FullLoader)

beijingAngleNum = y["beijingAngleNum"]
beijingPlanes = y["beijingPlanes"]
beijingSubDetectorSize = y["beijingSubDetectorSize"]
beijingVolumeSize = y["beijingVolumeSize"]
beijingParameterRoot = y["beijingParameterRoot"]
beijingSID = y["beijingSID"]
beijingSDD = y["beijingSDD"]
