
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int neurons;
double error,test_error,error1,train_data[120][15],test_data[60][15],mean_train[14],mean_test[14],variance_train[14],variance_test[14];
double **wgt_out,*hiden,**wgt_hidden,*output;

double activation(double f)
{
	double m=pow((1.0+exp(-1.0*f)),-1);
	return m;
}

double classify(int row,int data)
{
	int i,j,k;
	double sum,delta[3],val[3],true_value[3],class;
	int n=neurons;
	for(i=0;i<n;i++)
	{
		sum=0;
		for(j=0;j<14;j++)
		{			
				sum+=(test_data[row][j]*wgt_hidden[i][j]);
		}
		hiden[i]=(double)activation(sum);
		
	}
	for(j=0;j<3;j++)
	{
		sum=0;
		for(i=0;i<n;i++)
			sum+=hiden[i]*wgt_out[i][j];
		output[j]=(double)activation(sum);
		
	}
	val[0]=output[0];
	val[1]=output[1];
	val[2]=output[2];

	if(test_data[row][14]==1)
	{
		true_value[0]=1.0;
		true_value[1]=0.0;
		true_value[2]=0.0;
	}
	else if (test_data[row][14]==2)
	{
		true_value[0]=0.0;
		true_value[1]=1.0;
		true_value[2]=0.0;
	}
	else
	{
		true_value[0]=0.0;
		true_value[1]=0.0;
		true_value[2]=1.0;
	}
	if(row==0)
		error1=0.0;
	for(i=0;i<3;i++)
	{
		error1+=pow(true_value[i]-val[i],2);

		delta[i]=(true_value[i]-val[i])*(output[i]*(1-output[i]));

	}
	error1*=0.5;
	if(output[0]>output[1]&&output[0]>output[2])
		class=1.0;
	else if(output[1]>output[0]&&output[1]>output[2])
		class=2.0;
	else
		class=3.0;
	return class;
}

void backprop(int row)
{
	int i,j,k;
	double sum,delta[3],val[3],true_value[3];
	int n=neurons;
	for(i=0;i<n;i++)
	{
		sum=0;
		for(j=0;j<14;j++)
		{
			sum+=(train_data[row][j]*wgt_hidden[i][j]);
		}
		hiden[i]=(double)activation(sum);
	}
	for(j=0;j<3;j++)
	{
		sum=0;
		for(i=0;i<n;i++)
			sum+=hiden[i]*wgt_out[i][j];
		output[j]=(double)activation(sum);
		
	}
	val[0]=output[0];val[1]=output[1];val[2]=output[2];
	if(train_data[row][14]==1)
	{
		true_value[0]=1.0;
		true_value[1]=0.0;
		true_value[2]=0.0;
	}
	else if (train_data[row][14]==2)
	{
		true_value[0]=0.0;
		true_value[1]=1.0;
		true_value[2]=0.0;
	}
	else
	{
		true_value[0]=0.0;
		true_value[1]=0.0;
		true_value[2]=1.0;
	}
	if(row==0)
		error=0.0;
	for(i=0;i<3;i++)
	{
		error+=pow(true_value[i]-val[i],2);

		delta[i]=(true_value[i]-val[i])*(output[i]*(1-output[i]));
		for(j=0;j<n;j++)
		{
			wgt_out[j][i]=wgt_out[j][i]+(0.1*delta[i]*hiden[j]);
		}

	}
	error*=0.5;
	for(i=0;i<n;i++)
	{
		double p=0.0;
		for(j=0;j<3;j++)
		{
			p+=(delta[j]*wgt_out[i][j]*hiden[i]*(1-hiden[i]));
		}
		for(k=1;k<14;k++)
			wgt_hidden[i][k]+=(0.1*train_data[row][k]*p);
	}	



}
void gen_wgts(int n)
{
	
	int i,j;
	hiden=(double *)malloc(n*sizeof(double));
	output=(double *)malloc(3*sizeof(double));
	wgt_hidden = (double **)malloc(n* sizeof(double *));
    for (i=0; i<n; i++){
         wgt_hidden[i] = (double *)malloc(14*sizeof(double));
     }
    wgt_out = (double **)malloc(n* sizeof(double *));
    for (i=0; i<n; i++){
         wgt_out[i] = (double *)malloc(3*sizeof(double));
    }
    for(i=0;i<n;i++){
     	for(j=0;j<14;j++){
     		wgt_hidden[i][j]=2*((double)rand()/RAND_MAX-0.5);
     	}
     }	
     for(i=0;i<n;i++){
     	for(j=0;j<3;j++){
     		wgt_out[i][j]=2*((double)rand()/RAND_MAX-0.5);
     	}
     }

}
int main()
{
	srand((unsigned)time(NULL));
	double output_class,p_accuracy;
	int i,j,n,rs1=0,rs2=0,k,p_error[11],p_error1[11],min_error,best_no_neurons,best_no_neurons1,itr;
	FILE *f1,*f2,*f3,*f4,*f5;
	f1=fopen("train.csv","r");
	f2=fopen("test.csv","r");
	f3=fopen("iterations.csv","w");
	f4=fopen("accuracy.csv","w");
	f5=fopen("classification.csv","w");

	for(i=1;i<=13;i++){
		mean_train[i]=0;
	}
	i=0;
	while(fscanf(f1,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",&train_data[i][1],&train_data[i][2],&train_data[i][3],&train_data[i][4],&train_data[i][5],&train_data[i][6],&train_data[i][7],&train_data[i][8],&train_data[i][9],&train_data[i][10],&train_data[i][11],&train_data[i][12],&train_data[i][13],&train_data[i][14])!=EOF)
	{
		train_data[i][0]=1.0;
		for(j=1;j<=13;j++)
			mean_train[j]+=train_data[i][j];
		i++;
	}
	rs1=i;

	i=0;
	while(fscanf(f2,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",&test_data[i][1],&test_data[i][2],&test_data[i][3],&test_data[i][4],&test_data[i][5],&test_data[i][6],&test_data[i][7],&test_data[i][8],&test_data[i][9],&test_data[i][10],&test_data[i][11],&test_data[i][12],&test_data[i][13],&test_data[i][14])!=EOF)
	{
		test_data[i][0]=1.0;
		for(j=1;j<=13;j++)
			mean_test[j]+=test_data[i][j];
		i++;
	}
	rs2=i;
	/*Normalising test data*/
	for(j=1;j<=13;j++){          
		mean_test[j]/=rs2;
		mean_train[j]/=rs1;

	}
	for(i=0;i<rs2;i++){
		for(j=1;j<=13;j++){
			variance_test[j]+=pow(mean_test[j]-test_data[i][j],2);

		}
	}
	for(i=0;i<rs1;i++){
		for(j=1;j<=13;j++){
			variance_train[j]+=pow(mean_train[j]-train_data[i][j],2);
		}
	}
	for(j=1;j<=13;j++){
		variance_test[j]=pow(variance_test[j],0.5);
		variance_train[j]=pow(variance_train[j],0.5);
	}
	for(i=0;i<rs2;i++){
		for(j=1;j<=13;j++){
			test_data[i][j]=(test_data[i][j]-mean_test[j])/variance_test[j];
		}
	}
	for(i=0;i<rs1;i++){
		for(j=1;j<=13;j++){
			train_data[i][j]=(train_data[i][j]-mean_train[j])/variance_train[j];
		}
	}

	
	error=50;
	min_error=50;
	best_no_neurons=0;
	fprintf(f3, "no.of Neurons,Iterations\n");
	fprintf(f4, "no.of Neurons,Accuracy(%%)\n");
	for(neurons=10;neurons<=20;neurons++)
	{
		itr=0;
		gen_wgts(neurons);
		n=neurons;
		while(1)
		{
			itr++;
			if(error<=0.001)
			{
				error=100;
				break;
			}	
			for(k=0;k<rs1;k++)
			{
				backprop(k);
			}


		}
		p_error[neurons-10]=0;

		fprintf(f3, "%d,%d\n",n,itr);

		best_no_neurons1=0;
		min_error=100;
		p_error1[neurons-10]=0;
		for(j=0;j<rs2;j++)
		{

			output_class=classify(j,1);
			if(output_class!=test_data[j][14])
			{
				p_error1[neurons-10]+=1;
				
			}
		}
		p_accuracy=(1-(double)p_error1[neurons-10]/rs2)*100.0;
		fprintf(f4, "%d,%f\n",n,p_accuracy);
		if(p_error1[neurons-10]<min_error)
		{
			min_error=p_error1[neurons-10];
			best_no_neurons1=n;

		}
	}
	for(j=0;j<rs2;j++)
		{
			neurons=20;
			output_class=classify(j,1);
			fprintf(f5, "%f\n",output_class);
		}

	return 0;
}
