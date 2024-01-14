

# xyz_cm = [-29.37284854279016, 26.28042250735473, 17.50295293391675] # start
# xyz_cm = [-42.30380133083493, -1.8778051578191037, 31.922637565039345] # pouring
# xyz_cm = [-29.28116231426931, 26.284866616656334, 17.473099999999995] # human start
xyz_cm = [-42.46373558687616, -1.6768498291012008, 32.192309988228104] # human pour

print()
print(xyz_cm)
print()

# Human has a starting point at about -0.3, 0.25, 0.18
# Baxter has a starting point at about 0.9, -0.4, 0
# Human and Baxter are rotated 180 about z (x and y are negated)
xyz_m = [a/100 for a in xyz_cm]
xyz_m = [
  -xyz_m[0] + 0.6,
  -xyz_m[1] - 0.15,
  xyz_m[2] - 0.18,
  ]

print(xyz_m)
print()
