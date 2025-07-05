FROM openjdk:17-jdk-slim AS build

WORKDIR /app

COPY . .

RUN chmod +x ./gradlew

RUN ./gradlew build --no-daemon

FROM openjdk:17-jdk-slim

WORKDIR /app

COPY --from=build /app/build/libs/Gomoku-0.0.1-SNAPSHOT.jar app.jar

EXPOSE 10086

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
