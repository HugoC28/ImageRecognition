#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define N 3
#define nb_entrees_x 35
#define nb_entrees_y 25
#define nb_images 200 
#define nb_entrees 36
#define nb_cachees 23
#define nb_sorties 10

double wij[nb_entrees][nb_cachees];
double wjk[nb_cachees][nb_sorties];
double somme1[nb_cachees];
double somme2[nb_sorties];

double erreur,emin,eta=0.05;
double h[nb_cachees];
double s[nb_sorties];
double dc[nb_cachees];
double ds[nb_sorties];
double epsilon=0.005;

double sigmoid(double a){
    double alpha=0.8;
    return (1.0/(1.+exp(-alpha*a)));
} 

int main(){
	//Lecture base de données
	FILE *myFile;
	//double I[x][y][nb_images] = ...;
	//float fValues[nb_img][nb_entrees_y][nb_entrees_x];
	float I[nb_entrees_x][nb_entrees_y][nb_images];
	int n = 0;
	int i=0;
	int img=0;
	char fileName[20] = "BDD_img";
	strcat(fileName,".csv");
	myFile = fopen(fileName, "r");
	if (myFile == NULL) {
		printf("failed to open file\n");
		return 1;
	}
	while (fscanf(myFile, "%f", &I[n++][i][img]) == 1){
		fscanf(myFile, ",");
		if(n==nb_entrees_x){
			i++;
			n=0;
			if(i==nb_entrees_y){
				img++;
				i=0;
			}
		}
	}
	fclose(myFile);
	FILE *myFile2;
	//double sortie_desiree[N*N][nb_sorties] =  ... ;
	float sortie_desiree[nb_images][nb_sorties];
	n = 0;
	img=0;
	char fileName2[20] = "BDD_ans";
	strcat(fileName2,".csv");
	myFile2 = fopen(fileName2, "r");
	if (myFile2 == NULL) {
		printf("failed to open file\n");
		return 1;
	}
	while (fscanf(myFile2, "%f", &sortie_desiree[img][n++]) == 1) {
	fscanf(myFile2, ",");
		if(n==nb_sorties){
			img++;
			n=0;
		}
	}
	fclose(myFile2);

	register int j, t, k, cpt;
	emin=2000.;
	double err[nb_sorties];
	for(j=0; j<nb_cachees;j++){
		for ( i = 0; i < nb_entrees; i++){
	    		wij[i][j]=(double)(rand()%10)/10.;
		}
	}

	for(k=0; k<nb_sorties;k++){
		for ( j = 0; j < nb_cachees; j++){
	    		wjk[j][k]=(double)(rand()%10)/10.;
		}
	}

	//Création de nos masques
	int u,v,y,x;
 	double M[N*N][N*N];
	for(u=0;u<N;u++){
		for(v=0;v<N;v++){
	    		for(x=0;x<N;x++){
				for(y=0;y<N;y++){
		    			M[N*u+x][N*v+y]=cos(((2.*x+1.)*u*3.1415)/(2.*N))*cos(((2.*y+1.)*v*3.1415)/(2.*N));
				}
			}
		}
	}
	
	for (t=0;t<nb_images;t++){
		int c = nb_entrees_x;
		int l = nb_entrees_y;

		//double J[l][c][N*N]={{{0}}};
		double J[l][c][N*N];
		for (n=0;n<N*N;n++){
			for (i=0;i<l;i++){
				for(j=0;j<c;j++){
					J[i][j][n] = I[i][j][t];
				}
			}
		}
		while (c*l > nb_entrees){
			double Conv[l][c][N*N];
			//double Conv[l][c][N*N] = {{{0}}};
			for (n=0;n<N*N;n++){
				for (i=1;i<l-1;i++){
					for (j=1;j<c-1;j++){
						Conv[i][j][n] = M[n*N*N][n]*J[i-1][j-1][n] + M[n*N*N+1][n]*J[i-1][j][n] + M[n*N*N+2][n]*J[i-1][j+1][n] + M[n*N*N+3][n]*J[i][j-1][n] + M[n*N*N+4][n]*J[i][j][n] + M[n*N*N+5][n]*J[i][j+1][n] + M[n*N*N+6][n]*J[i+1][j-1][n] + M[n*N*N+7][n]*J[i+1][j][n] + M[n*N*N+8][n]*J[i+1][j+1][n];
					}
				}
			}

			double Relu[l][c][N*N];
			for (n=0;n<N*N;n++){
				for (i=0;i<l;i++){
					for (j=0;j<c;j++){
						if (Conv[i][j][n] < 0){	
							Relu[i][j][n] = - Conv[i][j][n];
						}
						else{
							Relu[i][j][n] = Conv[i][j][n];
						}
					}
				}
			}
			
			//double Pooling[l/2+1][c/2+1][N*N] = {{{0}}};
			double Pooling[l/2+1][c/2+1][N*N];
			int index;
			double max;
			for (n=0;n<N*N;n++){
				for (i=0;i<l/2;i++){
					for (j=0;j<c/2;j++){
						double Temp[4] = { Relu[2*i][2*j][n],Relu[2*i][2*j+1][n],Relu[2*i+1][2*j][n],Relu[2*i+1][2*j+1][n]};
						for (index=0;index<4;index++){
							if (Temp[index]>max){
								max=Temp[index];
							}
						}
						Pooling[i][j][n] = max;
						max=0.0;
					}
				}
			}
			l=l/2;
			c=c/2;
			if (c%2==1){
				c=c+1;
				l=l+1;
			}
			for (i=0;i<l;i++){
				for(j=0;j<c;j++){
					for (n=0;n<N*N;n++){
						J[i][j][n] = Pooling[i][j][n];
					}
				}
			}

		}
		double entree[N*N][nb_entrees] = {{0}};
		for (n=0;n<N*N;n++){
			for (i=0;i<l;i++){
				for (j=0;j<c;j++){
					entree[n][(i+1)*j]=J[i][j][n];
				}
			}
		}
		double sortie[N*N][nb_sorties];
	
		for(n=0;n<N*N;n++){
		    //forward
		    for(j = 0; j < nb_cachees; j++){

		        somme1[j]=0.;
		        for (i = 0; i < nb_entrees; i++){
		            somme1[j]+= wij[i][j]*entree[n][i];
		        }
		        h[j]=sigmoid(somme1[j]);
		    }

		    for(k = 0; k < nb_sorties; k++){

		        somme2[k]=0.;
		        for (j = 0; j < nb_cachees; j++){
		            somme2[k]+= wjk[j][k]*h[j];
		        }
		        s[k]=sigmoid(somme2[k]);
		    }

		    //erreur
		    err[n]=0;
		    for(k = 0; k < nb_sorties; k++){
		        err[n]+=(s[k]-sortie_desiree[n][k])*(s[k]-sortie_desiree[n][k]);
		    }

		    //backpropagation cachee
		    for(k = 0; k < nb_sorties; k++){
		        ds[k]=s[k]*(1.-s[k])*(sortie_desiree[n][k]-s[k]);
		    }

		    for(j = 0; j < nb_cachees; j++){
		        for(k = 0; k < nb_sorties; k++){
		            wjk[j][k]+=eta*ds[k]*h[j];
		        }
		    }

		    //backpropagation entree
		    for(j = 0; j < nb_cachees; j++){
		        dc[j]=0.;
		        for(k = 0; k < nb_sorties; k++){
		            dc[j]+=ds[k]*wjk[j][k];
		        }
		        dc[j]=h[j]*(1.-h[j])*dc[j];
		        
		    }

		    for(j = 0; j < nb_cachees; j++){
		        for(i = 0; i < nb_entrees; i++){
		            wij[i][j]+=eta*dc[j]*entree[n][i];
		        }
		    }

		}
		//calcul erreur
		erreur=0.;
		for(k = 0; k < nb_sorties; k++){
		    erreur+=err[k];
		}
		
		if(erreur<emin){
			emin=erreur;
		}
	}
	printf("emin :%d \n",emin);
	for (t=0;t<1;t++){
		int c = x;
		int l = y;

		//double J[l][c][N*N]={{{0}}};
		double J[l][c][N*N];
		for (n=0;n<N*N;n++){
			for (i=0;i<l;i++){
				for(j=0;j<c;j++){
					J[i][j][n] = I[i][j][t];
				}
			}
		}
		while (c*l > nb_entrees){
			double Conv[l][c][N*N];
			//double Conv[l][c][N*N] = {{{0}}};
			for (n=0;n<N*N;n++){
				for (i=1;i<l-1;i++){
					for (j=1;j<c-1;j++){
						Conv[i][j][n] = M[n*N*N][n]*J[i-1][j-1][n] + M[n*N*N+1][n]*J[i-1][j][n] + M[n*N*N+2][n]*J[i-1][j+1][n] + M[n*N*N+3][n]*J[i][j-1][n] + M[n*N*N+4][n]*J[i][j][n] + M[n*N*N+5][n]*J[i][j+1][n] + M[n*N*N+6][n]*J[i+1][j-1][n] + M[n*N*N+7][n]*J[i+1][j][n] + M[n*N*N+8][n]*J[i+1][j+1][n];
					}
				}
			}

			double Relu[l][c][N*N];
			for (n=0;n<N*N;n++){
				for (i=0;i<l;i++){
					for (j=0;j<c;j++){
						if (Conv[i][j][n] < 0){	
							Relu[i][j][n] = - Conv[i][j][n];
						}
						else{
							Relu[i][j][n] = Conv[i][j][n];
						}
					}
				}
			}
			
			//double Pooling[l/2+1][c/2+1][N*N] = {{{0}}};
			double Pooling[l/2+1][c/2+1][N*N];
			int index;
			double max;
			for (n=0;n<N*N;n++){
				for (i=0;i<l/2;i++){
					for (j=0;j<c/2;j++){
						double Temp[4] = { Relu[2*i][2*j][n],Relu[2*i][2*j+1][n],Relu[2*i+1][2*j][n],Relu[2*i+1][2*j+1][n]};
						for (index=0;index<4;index++){
							if (Temp[index]>max){
								max=Temp[index];
							}
						}
						Pooling[i][j][n] = max;
						max=0.0;
					}
				}
			}
			l=l/2;
			c=c/2;
			if (c%2==1){
				c=c+1;
				l=l+1;
			}
			for (i=0;i<l;i++){
				for(j=0;j<c;j++){
					for (n=0;n<N*N;n++){
						J[i][j][n] = Pooling[i][j][n];
					}
				}
			}

		}
		double entree[N*N][nb_entrees] = {{0}};
		for (n=0;n<N*N;n++){
			for (i=0;i<l;i++){
				for (j=0;j<c;j++){
					entree[n][(i+1)*j]=J[i][j][n];
				}
			}
		}
		double sortie[N*N][nb_sorties];
	
		for(n=0;n<N*N;n++){
		    //forward
		    for(j = 0; j < nb_cachees; j++){

		        somme1[j]=0.;
		        for (i = 0; i < nb_entrees; i++){
		            somme1[j]+= wij[i][j]*entree[n][i];
		        }
		        h[j]=sigmoid(somme1[j]);
		    }

		    for(k = 0; k < nb_sorties; k++){

		        somme2[k]=0.;
		        for (j = 0; j < nb_cachees; j++){
		            somme2[k]+= wjk[j][k]*h[j];
		        }
		        s[k]=sigmoid(somme2[k]);
		    }
		}
	}
	for (k=0;k<nb_sorties;k++){
		printf("%d",s[k]);
		printf("\n");
	}	
	return (0);	
}
