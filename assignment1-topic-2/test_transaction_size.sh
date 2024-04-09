javac A1_G5_t2.java

# test for size of data size

startSize=10
endSize=100
increment=10

for (( size=$startSize; size<=$endSize; size+=increment )); do
  java A1_G5_t2 "data/SMPF/${size}k.csv" 0.1
done