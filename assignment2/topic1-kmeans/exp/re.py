import subprocess
import os

# Java 실행 명령어와 인자 설정
java_command = "java"
java_class = "KMeansPlusPlusS"
input_file = "R15_output.csv"
output_file_prefix = "clusters_output"

# 반복 횟수 설정
num_runs = 100

# 저장 경로 설정
output_directory = "/Users/geumrin/Downloads/A2/KMEANS-java/results"

# 디렉토리 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Java 프로그램 100번 실행
for i in range(1, num_runs + 1):
    # 각 실행 결과를 저장할 파일 이름 설정
    output_file = os.path.join(output_directory, f"{output_file_prefix}_{i}.txt")
    
    # Java 프로그램 실행
    with open(output_file, "w") as f:
        subprocess.run([java_command, java_class, input_file], stdout=f, stderr=subprocess.STDOUT)
    
    print(f"Run {i} completed and saved to {output_file}")

print("All runs completed.")
