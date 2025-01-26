## Quick copy paste list over sumo sim generation

Sumo location:
C:\Program Files (x86)\Eclipse\Sumo

Tools:
NetEdit:
C:\Program Files (x86)\Eclipse\Sumo\bin\netedit.exe

NetGenerate:
C:\Program Files (x86)\Eclipse\Sumo\bin\netgenerate.exe

OSMWebWizzard:
C:\Program Files (x86)\Eclipse\Sumo\tools\osmWebWizzard.py

RandomTrips:
C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py

Sumo data dir:
Project/data/sumo/

Generate random traffic:

### This has to much traffic, need to train on less aswell

python randomTrips.py -n C:/Users/pc/Documents/Trafic/data/sumo/Sim04/osm.net.xml -r C:/Users/pc/Documents/Trafic/data/sumo/Sim04/routes2.rou.xml -b 0 -e 10000 -l -p 1

# <route-files value="osm.bus.trips.xml,osm.passenger.trips.xml"/> standard

sumo-gui -c osm.sumocfg --step-length 0.1 --random --step-method.ballistic --time-to-teleport -1 --waiting-time-memory 500

training between p 1 - 2.5 (2.5 a bit slow. can also be a tiny bit more then 1)
