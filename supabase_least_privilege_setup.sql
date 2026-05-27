-- Supabase production bootstrap for historical_evaluator.
-- Run this script with an admin role (for example, postgres) in SQL Editor.
-- Then use role historical_app in DATABASE_URL with DB_AUTO_MIGRATE=false.

begin;

-- 1) Create or rotate application role with least privileges.
do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'historical_app') then
        create role historical_app
            login
            password 'CHANGE_ME_STRONG_PASSWORD'
            nosuperuser
            nocreatedb
            nocreaterole
            noinherit
            noreplication;
    else
        alter role historical_app with
            login
            password 'CHANGE_ME_STRONG_PASSWORD'
            nosuperuser
            nocreatedb
            nocreaterole
            noinherit
            noreplication;
    end if;
end
$$;

alter role historical_app set search_path = public;

-- 2) Ensure schema/tables exist.
create table if not exists public.text_evaluations (
    id uuid primary key,
    created_at timestamptz not null default now(),
    source_text text not null
);

create table if not exists public.text_candidate_results (
    id uuid primary key,
    text_evaluation_id uuid not null references public.text_evaluations(id) on delete cascade,
    slot smallint not null,
    label text not null,
    candidate_text text not null,
    bleu_score double precision not null,
    bert_precision double precision not null,
    bert_recall double precision not null,
    bert_f1 double precision not null
);

create table if not exists public.image_evaluations (
    id uuid primary key,
    created_at timestamptz not null default now(),
    prompt_text text not null,
    fid_1_vs_23 double precision not null,
    fid_2_vs_13 double precision not null,
    fid_3_vs_12 double precision not null,
    clip_1 double precision not null,
    clip_2 double precision not null,
    clip_3 double precision not null
);

create table if not exists public.image_assets (
    id uuid primary key,
    image_evaluation_id uuid not null references public.image_evaluations(id) on delete cascade,
    slot smallint not null,
    filename text not null,
    mime_type text not null,
    sha256 text not null,
    image_bytes bytea not null
);

create table if not exists public.text_expert_reviews (
    id uuid primary key,
    text_evaluation_id uuid not null references public.text_evaluations(id) on delete cascade,
    created_at timestamptz not null default now(),
    evaluator_name text not null,
    evaluator_specialty text not null,
    evaluator_institution text not null,
    observations text null,
    responses_json text not null
);

create table if not exists public.image_expert_reviews (
    id uuid primary key,
    image_evaluation_id uuid not null references public.image_evaluations(id) on delete cascade,
    created_at timestamptz not null default now(),
    evaluator_name text not null,
    evaluator_specialty text not null,
    evaluator_institution text not null,
    observations text null,
    responses_json text not null
);

-- 3) RLS required by Supabase linter.
alter table public.text_evaluations enable row level security;
alter table public.text_candidate_results enable row level security;
alter table public.image_evaluations enable row level security;
alter table public.image_assets enable row level security;
alter table public.text_expert_reviews enable row level security;
alter table public.image_expert_reviews enable row level security;

-- 4) Restrict app role to required objects only.
grant connect on database postgres to historical_app;
grant usage on schema public to historical_app;
revoke create on schema public from historical_app;

grant select, insert, update, delete on table public.text_evaluations to historical_app;
grant select, insert, update, delete on table public.text_candidate_results to historical_app;
grant select, insert, update, delete on table public.image_evaluations to historical_app;
grant select, insert, update, delete on table public.image_assets to historical_app;
grant select, insert, update, delete on table public.text_expert_reviews to historical_app;
grant select, insert, update, delete on table public.image_expert_reviews to historical_app;

-- 5) Policies that allow only historical_app to read/write through RLS.
drop policy if exists text_evaluations_app_rw on public.text_evaluations;
create policy text_evaluations_app_rw
on public.text_evaluations
for all
to historical_app
using (true)
with check (true);

drop policy if exists text_candidate_results_app_rw on public.text_candidate_results;
create policy text_candidate_results_app_rw
on public.text_candidate_results
for all
to historical_app
using (true)
with check (true);

drop policy if exists image_evaluations_app_rw on public.image_evaluations;
create policy image_evaluations_app_rw
on public.image_evaluations
for all
to historical_app
using (true)
with check (true);

drop policy if exists image_assets_app_rw on public.image_assets;
create policy image_assets_app_rw
on public.image_assets
for all
to historical_app
using (true)
with check (true);

drop policy if exists text_expert_reviews_app_rw on public.text_expert_reviews;
create policy text_expert_reviews_app_rw
on public.text_expert_reviews
for all
to historical_app
using (true)
with check (true);

drop policy if exists image_expert_reviews_app_rw on public.image_expert_reviews;
create policy image_expert_reviews_app_rw
on public.image_expert_reviews
for all
to historical_app
using (true)
with check (true);

commit;
