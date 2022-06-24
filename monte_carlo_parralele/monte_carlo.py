#! /bin/python3

from mpi4py import MPI
comm = MPI.COMM_WORLD
name = MPI . Get_processor_name ()
rank = comm.Get_rank()
nb_proc = comm.Get_size()


import numpy as np
import random
import time


nx = 50
ny = 50
nb_tirages = 60
valeurs_aux_bords = np.array([0.0,1.0,2.0,3.0])  # valeurs de la solution sur les bords y=0, x=nx, y=ny et x=0

def conditions_aux_bords(grille):  # Initialisation de la solution sur les bords
	for i in range(0,nx): # y = 0
		grille[i][0] = valeurs_aux_bords[0]

	for i in range(0,ny): # x = nx
		grille[nx-1][i] = valeurs_aux_bords[1]

	for i in range(0,nx): # y = ny
		grille[i][ny-1] = valeurs_aux_bords[2]

	for i in range(0,ny): # x = 0
		grille[0][i] = valeurs_aux_bords[3]

def calcul_solution(grille,deb,fin):  #On calcule la solution locale, délimitée par les indices de lignes deb et fin
	random.seed(time.gmtime(deb*fin*deb*fin))
	for j in range(max(deb,1),fin):
		for i in range(1,nx-1):
			for n in range(1,nb_tirages):
				pos_x = i  # initialisation de la position x de la particule
				pos_y = j  # initialisation de la position y de la particule
				valeur = 0.0
				stop = 0   # vaudra 1 si un bord est atteint

				while stop != 1:
					decision = random.randrange(2)  # 0 ou 1
					if decision == 0:
						pos_x+=1
					else:
						pos_x-=1

					decision = random.randrange(2)  # 0 ou 1
					if decision == 1:
						pos_y+=1
					else:
						pos_y-=1

					if (pos_x == 0) | (pos_x == nx-1) | (pos_y == 0) | (pos_y == ny-1):
						valeur = grille[pos_x][pos_y]
						stop = 1

				grille[i][j] += valeur
			grille[i][j] /= nb_tirages 
	TAG1=10
	if rank != 0:
		comm.send(grille[:,deb:(fin)], dest = 0,tag= TAG1) #On transmet au thread 0 les lignes calculées par ce thread
	else:
		for k in range(1,nb_proc):
			deb_k = k*(ny-1)//nb_proc  #debut
			fin_k = (k+1)*(ny-1)//nb_proc  #fin
			grille[:,deb_k:(fin_k)]=comm.recv(source = k, tag = TAG1)
	

def ecriture(grille):
	fichier = open("monte_carlo.vtk","w")
	Nbnoe = nx*ny
	dx = 1.0/(nx-1);
	dy = 1.0/(ny-1);

	fichier.write("# vtk DataFile Version 2.0\n")
	fichier.write("Laplacien stochastique\n")
	fichier.write("ASCII\n")
	fichier.write("DATASET STRUCTURED_POINTS\n")
	fichier.write("DIMENSIONS %d %d 1\n" %(nx,ny))
	fichier.write("ORIGIN 0 0 0\n")
	fichier.write("SPACING %f %f 1\n" %(dx,dy))
	fichier.write("POINT_DATA %d\n" % Nbnoe)
	fichier.write("SCALARS Concentration float\n")
	fichier.write("LOOKUP_TABLE default\n")

	for i in range(0,nx):
		for j in range(0,ny):
			fichier.write("%f\n" % grille[i][j])
	fichier.close();			


def partitionnement(ny):  #On partitionne nos lignes suivant le nombre de coeurs
	deb = rank*ny//nb_proc  #debut
	fin = (rank+1)*ny//nb_proc  #fin
	return [deb,fin]


def main():
	comm.Barrier()
	if rank==0:
		t0 = MPI.Wtime()
	grille = np.zeros((nx,ny),dtype=float) # Initialisation de la solution
	conditions_aux_bords(grille)
	[deb,fin]=partitionnement(ny-1)
	print(f'Machine: {name}, process rank: {rank} sur {nb_proc}. deb={deb},fin={fin}\n')
	calcul_solution(grille,deb,fin)
	if rank==0:
		t1=MPI.Wtime()
		print(str(nb_proc)+ " processeurs, Temps d'exécution: "+str(t1-t0))
		ecriture(grille)


main()