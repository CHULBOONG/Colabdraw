import cv2, random, os, sys
import numpy as np
from copy import deepcopy
# 원본 이미지와 생성한 이미지 비교
from skimage.measure import compare_mse
#병렬 처리
import multiprocessing as mp

#파일 불러오기
filepath = sys.argv[1]
filename, ext = os.path.splitext(os.path.basename(filepath))

#이미지를 bgr 형식으로 읽어오기
img = cv2.imread(filepath)
#이미지 크기들 각각 저장해둠
height, width, channels = img.shape

# hyperparameters(초매개변수)
# 첫세대 유전자 갯수
n_initial_genes = 50 
#한 세대당 유전자 그룹 숫자
n_population = 50
#돌연변이 발생 확률
prob_mutation = 0.01
#유전자 그룹에 원 추가될 확률
prob_add = 0.3
#유전자 그룹에 원 삭제될 확률
prob_remove = 0.2

#원 크기, 100회마다 저장
min_radius, max_radius = 5, 15
save_every_n_iter = 100

# Gene(유전자(동그라미) 한개에 대한 클래스)
class Gene():
  def __init__(self):
    #xy좌표 0에서 너비나 높이만큼 범위로 지정
    self.center = np.array([random.randint(0, width), random.randint(0, height)])
    #반지름
    self.radius = random.randint(min_radius, max_radius)
    #색
    self.color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

#돌연변이 정의
  def mutate(self):
    #(가우시안분포) 평균 15 표준편차 4인 분포에서 랜덤 숫자 추출/100으로나눔
    #즉 평균 플러스 마이너스 15% 정도만큼 변이시킴
    mutation_size = max(1, int(round(random.gauss(15, 4)))) / 100

    r = random.uniform(0, 1)
    # 33%확률로 반지름을 바꿈
    if r < 0.33: # radius
      self.radius = np.clip(random.randint(
        int(self.radius * (1 - mutation_size)),
        int(self.radius * (1 + mutation_size))
      ), 1, 100)
      # 33%의 확률로 중심의 위치를 바꿈
    elif r < 0.66: # center
      self.center = np.array([
        np.clip(random.randint(
          int(self.center[0] * (1 - mutation_size)),
          int(self.center[0] * (1 + mutation_size))),
        0, width),
        np.clip(random.randint(
          int(self.center[1] * (1 - mutation_size)),
          int(self.center[1] * (1 + mutation_size))),
        0, height)
      ])
      # 33% 확률로 색을 바꿈
    else: # color
      self.color = np.array([
        np.clip(random.randint(
          int(self.color[0] * (1 - mutation_size)),
          int(self.color[0] * (1 + mutation_size))),
        0, 255),
        np.clip(random.randint(
          int(self.color[1] * (1 - mutation_size)),
          int(self.color[1] * (1 + mutation_size))),
        0, 255),
        np.clip(random.randint(
          int(self.color[2] * (1 - mutation_size)),
          int(self.color[2] * (1 + mutation_size))),
        0, 255)
      ])

# compute fitness 얼마나 맞는지 판단
def compute_fitness(genome):
  #numpy.ones() : 1로 채워진 배열 생성, 동일 사이즈의 하얀 배경 생성
  out = np.ones((height, width, channels), dtype=np.uint8) * 255

  for gene in genome:
    #OpenCV의 circle 함수 사용, 중심 반지름 색깔 정보 이용
    #thickness=-1 원 색 채우기
    cv2.circle(out, center=tuple(gene.center), radius=gene.radius, color=(int(gene.color[0]), int(gene.color[1]), int(gene.color[2])), thickness=-1)

  # mean squared error
  # compare_mse()로 불러온 이미지와의 차이(mse)를 계산
  # fitness는 높을수록 좋고, mse는 낮을수록 좋다 = 그만큼 원본과 비슷하다
  fitness = 255. / compare_mse(img, out)
  #반환
  return fitness, out

# compute population
def compute_population(g):
  genome = deepcopy(g)
  # mutation 한꺼번에 변이시키기
  if len(genome) < 200:
    for gene in genome:
      if random.uniform(0, 1) < prob_mutation:
        gene.mutate()
  else:
    #random.sample(a,k) a에서 k개만큼 랜덤추출
    for gene in random.sample(genome, k=int(len(genome) * prob_mutation)):
      gene.mutate()

  # add gene 처음 50개에서 하나씩 더 늘어남
  if random.uniform(0, 1) < prob_add:
    genome.append(Gene())

  # remove gene 삭제
  if len(genome) > 0 and random.uniform(0, 1) < prob_remove:
    genome.remove(random.choice(genome))

  # compute fitness
  new_fitness, new_out = compute_fitness(genome)

  return new_fitness, genome, new_out

# main
if __name__ == '__main__':
  os.makedirs('result', exist_ok=True)
# cpu 갯수를 센 후 한 개 적은 멀티프로세싱 풀 이용
  p = mp.Pool(mp.cpu_count() - 1)

  # 1st gene
  #변수로 첫번째 유전자 생성
  best_genome = [Gene() for _ in range(n_initial_genes)]

  best_fitness, best_out = compute_fitness(best_genome)

  n_gen = 0

  while True:
    try:
      #[deepcopy(best_genome)] * n_population 을 compute_population로 넘겨줌
      results = p.map(compute_population, [deepcopy(best_genome)] * n_population)
    except KeyboardInterrupt:
      p.close()
      break

    results.append([best_fitness, best_genome, best_out])
#새로운 유전 평가
    new_fitnesses, new_genomes, new_outs = zip(*results)

#내림차순 정렬
    best_result = sorted(zip(new_fitnesses, new_genomes, new_outs), key=lambda x: x[0], reverse=True)

    best_fitness, best_genome, best_out = best_result[0]

    # end of generation 세대 종료, 프린트
    print('Generation #%s, Fitness %s' % (n_gen, best_fitness))
    n_gen += 1

    # visualize
    if n_gen % save_every_n_iter == 0:
      # cv2.imwrite() 가장 좋은 아웃풋을 이미지를 저장
      cv2.imwrite('result/%s_%s.jpg' % (filename, n_gen), best_out)

    # cv2.imshow('best out', best_out)
    if cv2.waitKey(1) == ord('q'):
     p.close()
     break
# 이미지를 윈도우에 보여줌 colab에선 안쓸거임
  # cv2.imshow('best out', best_out)
  cv2.waitKey(0)
