import os

# 입력 디렉토리와 파일 설정
input_directory = "/Users/geumrin/Downloads/A2/KMEANS-java/results"
output_file_prefix = "clusters_output"
k_estimates_file = "k_estimates.txt"

# k 추정값을 저장할 파일 열기
with open(k_estimates_file, "w") as k_estimates:
    # 각 클러스터링 결과 파일에 대해 처리
    for i in range(1, 101):
        cluster_output_file = os.path.join(input_directory, f"{output_file_prefix}_{i}.txt")
        
        with open(cluster_output_file, "r") as f:
            lines = f.readlines()
        
        # 첫 줄을 k_estimates_file에 저장
        k_estimates.write(f"{lines[0].strip()} for {output_file_prefix}_{i}.txt\n")
        
        # 나머지 줄을 다시 원래 파일에 저장
        with open(cluster_output_file, "w") as f:
            f.writelines(lines[1:])
        
        print(f"Processed {cluster_output_file}")

print("All files have been processed and k estimates have been saved to k_estimates.txt.")
