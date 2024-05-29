import subprocess
import os

# Java 실행 명령어와 인자 설정
java_command = "java"
java_class = "CalculateMetrics"
input_file = "R15_output.csv"
output_directory = "/Users/geumrin/Downloads/A2/KMEANS-java/results"
output_file_prefix = "clusters_output"
results_file = "metrics_results.txt"

# 결과를 저장할 파일을 열기
with open(results_file, "w") as results:
    # 각 클러스터링 결과 파일에 대해 Java 프로그램 실행
    for i in range(1, 101):
        cluster_output_file = os.path.join(output_directory, f"{output_file_prefix}_{i}.txt")
        
        # Java 프로그램 실행
        result = subprocess.run([java_command, java_class, input_file, cluster_output_file],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 실행 결과 저장
        results.write(f"Results for {cluster_output_file}:\n")
        results.write(result.stdout)
        results.write(result.stderr)
        results.write("\n" + "="*50 + "\n\n")
        
        print(f"Processed {cluster_output_file}")

print("All results have been processed and saved to metrics_results.txt.")
